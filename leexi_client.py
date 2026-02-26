import logging
import time

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import (
    CORMA_EMAIL_DOMAIN,
    CORMA_TEAM_NAMES,
    FEEDBACK_PROMPT_TITLE,
    LEEXI_BASE_URL,
    LEEXI_PAGE_SIZE,
    LEEXI_RATE_LIMIT_PER_MIN,
)
from models import CallMetadata, Speaker

logger = logging.getLogger("corma-feedback.leexi")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class LeexiClient:
    def __init__(self, key_id: str, key_secret: str) -> None:
        self.session = requests.Session()
        self.session.auth = (key_id, key_secret)
        self._request_timestamps: list[float] = []

    def _throttle(self) -> None:
        """Sleep if approaching the 50 req/min rate limit."""
        now = time.time()
        window_start = now - 60
        self._request_timestamps = [
            t for t in self._request_timestamps if t > window_start
        ]
        if len(self._request_timestamps) >= LEEXI_RATE_LIMIT_PER_MIN - 1:
            oldest = self._request_timestamps[0]
            sleep_time = 60 - (now - oldest) + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limit throttle: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def _request(self, method: str, endpoint: str, params: dict | None = None) -> dict:
        self._throttle()
        url = f"{LEEXI_BASE_URL}/{endpoint.lstrip('/')}"
        logger.debug(f"{method} {url} params={params}")
        resp = self.session.request(method, url, params=params, timeout=30)
        self._request_timestamps.append(time.time())
        resp.raise_for_status()
        return resp.json()

    def get_calls(self, from_date: str, to_date: str) -> list[dict]:
        """Fetch all calls in the date range, handling pagination."""
        all_calls: list[dict] = []
        page = 1

        while True:
            params = {
                "date_filter": "performed_at",
                "from": from_date,
                "to": to_date,
                "with_simple_transcript": "true",
                "items": LEEXI_PAGE_SIZE,
                "page": page,
                "order": "performed_at desc",
            }
            data = self._request("GET", "/calls", params=params)
            calls = data.get("data", [])
            all_calls.extend(calls)
            logger.info(
                f"Fetched page {page}: {len(calls)} calls "
                f"(total so far: {len(all_calls)})"
            )
            if len(calls) < LEEXI_PAGE_SIZE:
                break
            page += 1

        return all_calls

    def get_call_details(self, uuid: str) -> dict:
        """Fetch full details for a single call."""
        data = self._request("GET", f"/calls/{uuid}")
        return data.get("data", data)

    def extract_prompt_output(self, call_detail: dict) -> tuple[str | None, bool]:
        """Extract the 'product feedback prompt' output from a call's prompts.

        Returns:
            (prompt_output_text, has_feedback_prompt)
        """
        prompts = call_detail.get("prompts", [])
        if not prompts:
            return None, False

        for prompt in prompts:
            title = (prompt.get("title") or "").strip().lower()
            if title == FEEDBACK_PROMPT_TITLE:
                completions = prompt.get("completions", [])
                completion_text = "\n".join(completions) if completions else ""
                if completion_text.strip():
                    return completion_text.strip(), True
                return None, True  # Prompt exists but empty

        return None, False

    def build_call_metadata(
        self,
        raw_call: dict,
        feedback_prompt_output: str | None = None,
        has_feedback_prompt: bool = False,
    ) -> CallMetadata:
        """Normalize a raw Leexi call dict into a CallMetadata model."""
        speakers = self._build_speakers(raw_call.get("speakers", []))
        transcript = self._build_transcript_text(raw_call, speakers)

        leexi_url = raw_call.get("leexi_url")

        # Extract company name and contact email from Leexi deal/contact info
        company_name, contact_email = self._extract_company_info(raw_call, speakers)

        # Extract call type from deal stage
        call_type = self._extract_call_type(raw_call)

        # Extract owner (Corma team member) from Leexi
        owner_name = self._extract_owner(raw_call)

        return CallMetadata(
            uuid=raw_call.get("uuid", ""),
            title=raw_call.get("title"),
            duration=raw_call.get("duration"),
            performed_at=raw_call.get("performed_at"),
            direction=raw_call.get("direction"),
            speakers=speakers,
            customer_emails=raw_call.get("customer_email_addresses", []),
            transcript_text=transcript,
            leexi_url=leexi_url,
            feedback_prompt_output=feedback_prompt_output,
            has_feedback_prompt=has_feedback_prompt,
            leexi_company_name=company_name,
            leexi_contact_email=contact_email,
            leexi_call_type=call_type,
            leexi_owner_name=owner_name,
        )

    # Mapping from Leexi deal stage names to our 4 allowed call types.
    # Leexi stages look like: "1 Discovery & Qualification", "2. Demo",
    # "3. POC / Pilot", "4. Negotiation", "5. Won", "6. Lost" etc.
    STAGE_TO_CALL_TYPE: dict[str, str] = {
        "discovery": "Discovery",
        "qualification": "Discovery",
        "demo": "Demo",
        "demonstration": "Demo",
        "poc": "Demo",
        "pilot": "Demo",
        "negotiation": "Discovery",
        "won": "Customer Success",
        "customer": "Customer Success",
        "onboarding": "Customer Success",
        "success": "Customer Success",
        "support": "Support",
        "lost": "Discovery",
    }

    def _extract_call_type(self, raw_call: dict) -> str | None:
        """Derive call type from Leexi deal stage and call title.

        Maps to one of: Demo, Customer Success, Discovery, Support.
        Falls back to title-based heuristics if no deal stage.
        """
        # 1. Try deal.stage.name
        deal = raw_call.get("deal")
        if deal and isinstance(deal, dict):
            stage = deal.get("stage")
            if stage and isinstance(stage, dict):
                stage_name = (stage.get("name") or "").lower()
                # Check each keyword in the stage name against our mapping
                for keyword, call_type in self.STAGE_TO_CALL_TYPE.items():
                    if keyword in stage_name:
                        return call_type

        # 2. Fallback: infer from call title
        title = (raw_call.get("title") or "").lower()
        title_hints = {
            "demo": "Demo",
            "démo": "Demo",
            "démonstration": "Demo",
            "demonstration": "Demo",
            "discovery": "Discovery",
            "découverte": "Discovery",
            "qualification": "Discovery",
            "rencontre": "Discovery",
            "follow-up": "Customer Success",
            "follow up": "Customer Success",
            "suivi": "Customer Success",
            "checkup": "Customer Success",
            "check-up": "Customer Success",
            "onboarding": "Customer Success",
            "retour": "Customer Success",
            "support": "Support",
        }
        for hint, call_type in title_hints.items():
            if hint in title:
                return call_type

        return None

    def _extract_owner(self, raw_call: dict) -> str | None:
        """Extract the Corma team member who owns this call from Leexi."""
        owner = raw_call.get("owner")
        if owner and isinstance(owner, dict):
            name = owner.get("name")
            if name:
                return name
        return None

    def _extract_company_info(
        self, raw_call: dict, speakers: list[Speaker]
    ) -> tuple[str | None, str | None]:
        """Extract company name and primary contact email from Leexi call data.

        Sources (in priority order):
        1. deal.name — strip common suffixes like "New deal", "New Deal", "- New Deal"
        2. External speaker email domain — capitalize the domain name
        3. customer_email_addresses — use the first non-Corma email domain
        """
        import re

        company_name: str | None = None
        contact_email: str | None = None

        # 1. Try deal name
        deal = raw_call.get("deal")
        if deal and isinstance(deal, dict):
            deal_name = deal.get("name", "")
            if deal_name:
                # Strip common suffixes: "RDMC New deal", "ITESOFT - New Deal", etc.
                cleaned = re.sub(
                    r'\s*[-–—]?\s*[Nn]ew\s*[Dd]eal\s*$', '', deal_name
                ).strip()
                if cleaned:
                    company_name = cleaned

        # 2. Find the first external contact email
        customer_emails = raw_call.get("customer_email_addresses", [])
        for email in customer_emails:
            if email and not email.lower().endswith(f"@{CORMA_EMAIL_DOMAIN}"):
                contact_email = email
                break

        # 3. If no company name from deal, try to infer from email domain
        if not company_name and contact_email:
            domain = contact_email.split("@")[-1].split(".")[0]
            company_name = domain.upper() if len(domain) <= 5 else domain.capitalize()

        # 4. If still no contact email, check external speakers
        if not contact_email:
            for speaker in speakers:
                if not speaker.is_corma_team and speaker.email:
                    contact_email = speaker.email
                    break

        return company_name, contact_email

    def _build_speakers(self, raw_speakers: list[dict]) -> list[Speaker]:
        speakers: list[Speaker] = []
        for s in raw_speakers:
            email = s.get("email_address")
            name = s.get("name", "Unknown")
            is_corma = self._is_corma_member(name, email, s.get("is_user", False))
            speakers.append(
                Speaker(
                    name=name,
                    email=email,
                    is_corma_team=is_corma,
                    speaker_index=s.get("index", 0),
                )
            )
        return speakers

    @staticmethod
    def _is_corma_member(name: str, email: str | None, is_user: bool) -> bool:
        """Determine if a speaker is a Corma team member.

        Priority: is_user flag > email domain > first-name match.
        """
        if is_user:
            return True
        if email and email.lower().endswith(f"@{CORMA_EMAIL_DOMAIN}"):
            return True
        first_name = name.strip().split()[0].lower() if name else ""
        return first_name in CORMA_TEAM_NAMES

    def _build_transcript_text(
        self, raw_call: dict, speakers: list[Speaker]
    ) -> str | None:
        """Get transcript as readable text.

        Prefer simple_transcript; fall back to structured transcript.
        """
        simple = raw_call.get("simple_transcript")
        if simple and simple.strip():
            return simple.strip()

        structured = raw_call.get("transcript")
        if not structured:
            return None

        speaker_map = {s.speaker_index: s.name for s in speakers}
        lines: list[str] = []
        for segment in structured:
            idx = segment.get("speaker_index", 0)
            speaker_name = speaker_map.get(idx, f"Speaker {idx}")
            words = " ".join(
                item.get("content", "") for item in segment.get("items", [])
            )
            if words.strip():
                lines.append(f"[{speaker_name}]: {words.strip()}")
        return "\n".join(lines) if lines else None
