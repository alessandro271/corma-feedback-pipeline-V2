import json
import logging
import re

import anthropic

from config import CLAUDE_MODEL, MAX_TRANSCRIPT_CHARS
from models import CallAnalysisResult, CallMetadata, FeedbackItem

logger = logging.getLogger("corma-feedback.analyzer")

SYSTEM_PROMPT = """\
You are a product feedback analyst for Corma, a SaaS company that helps \
businesses manage their software access and licenses. You analyze sales call \
data to extract actionable product feedback for the product team.

Your goal is to identify ONLY features and capabilities that Corma does NOT \
currently have, so the product team can prioritise building them. \
Quality over quantity — only include feedback where you have strong confidence \
that it represents a genuine product gap.

You will receive TWO sources of information:
1. PRE-EXTRACTED FEEDBACK from Leexi AI — structured feedback points already identified \
from the call, including need categorization (Blocking, Must-have, Nice-to-have).
2. The FULL TRANSCRIPT of the call — for context, verification, and additional insights.

Your task:
1. Classify the call type as one of: Demo, Customer Success, Discovery, Support, Other. \
Use these definitions:
   - "Demo" = product demonstration or POC/pilot calls
   - "Discovery" = initial discovery, qualification, or sales negotiation calls
   - "Customer Success" = follow-up, onboarding, check-in, or feedback calls with existing customers
   - "Support" = technical support or issue resolution calls
   - "Other" = anything that doesn't fit the above (internal meetings, etc.)
2. Identify which participants are Corma team members and which are customers/prospects.
3. Use the pre-extracted feedback as your STARTING POINT, but critically evaluate each one.
4. For EVERY potential feedback point, carefully check the transcript for evidence that \
the feature ALREADY EXISTS in Corma. You MUST EXCLUDE a feedback point if:
   - The Corma sales rep explains how Corma already does this \
(e.g. "yes, we can do that", "we already have this", "let me show you how it works")
   - The Corma rep demonstrates the feature during the call
   - The Corma rep confirms the capability exists \
(e.g. "we collect last activity data and can help revoke inactive licenses")
   - The feedback describes a general use case that Corma's existing product already covers \
(e.g. "visibility into software licenses" — this IS the core product)
5. For feedback points that pass the filter (genuine product gaps), cross-reference with \
the transcript to:
   - Confirm the Corma rep says things like "we don't currently have this", \
"that's not available yet", "it's on our roadmap", or stays silent/deflects on the topic
   - Extract an exact verbatim quote from the CUSTOMER (not the Corma rep)
6. Extract any ADDITIONAL specific product gaps from the transcript that the pre-extraction missed.
7. Extract the potential MRR value and company info from the pre-extracted output header.
8. Extract the company size (number of users) and company domain (e.g. "smart-trade.net") \
from the pre-extracted output header if available.

CRITICAL RULES — What to INCLUDE vs EXCLUDE:

INCLUDE (genuine product gaps):
- Specific integrations Corma doesn't have yet \
(e.g. "Integration with Employment Hero HR platform")
- Specific features confirmed as not existing \
(e.g. "API for triggering license changes programmatically")
- Specific automation workflows that don't exist yet \
(e.g. "Automated SMS-based 2FA handling for browser agents")
- Concrete capabilities the Corma rep acknowledges are missing or coming later

EXCLUDE (these are NOT useful product feedback):
- Features the Corma rep explains or demonstrates during the call
- General descriptions of what the customer wants Corma for \
(e.g. "wants better visibility into SaaS spend" — too vague and core product)
- Capabilities the Corma rep confirms already exist, even if the customer phrases them as needs
- Vague or high-level statements \
(e.g. "needs automation" or "wants cost optimization" — not actionable)
- Observations about the customer's current situation or pain points that aren't feature requests
- Feedback about pricing, contracts, or commercial terms (unless about a specific product capability)

CATEGORY CLASSIFICATION:
Each feedback item must be classified into exactly ONE of these categories based on its nature:
- "Feature Request" = a new capability or feature Corma doesn't have (e.g. "automated cost owner detection")
- "Integration Request" = a request to integrate with a specific third-party tool or platform (e.g. "Integration with Employment Hero")
- "Bug Report" = something that is broken or not working as expected
- "UX Issue" = usability or user experience problem with existing features
- "Performance Issue" = slowness, latency, or scaling problems
- "Pricing Feedback" = feedback about pricing model, tiers, or cost
- "Other" = doesn't fit any of the above
Do NOT default to "Integration Request" or "Other" — carefully consider the nature of each item.

SPECIFICITY REQUIREMENT:
Each feedback item MUST describe a specific, concrete feature or integration to build. \
Ask yourself: "Could a product manager create a Jira ticket directly from this?" \
If the answer is no because it's too vague, do NOT include it.

BAD (too vague): "Enhanced visibility into orphan accounts and unused licenses"
GOOD (specific): "Integration with Employment Hero (HR platform)"
BAD (too vague): "Automated SaaS management capabilities"
GOOD (specific): "API endpoint to programmatically trigger license quantity changes on ION platform"

Guidelines:
- Only extract feedback from customer/prospect statements, not from Corma team members.
- verbatim_quote must be exact text from the CUSTOMER in the transcript \
(copy-paste, do not paraphrase).
- If no genuine product gaps are found, return an empty feedback_items list. \
An empty list is better than low-quality feedback.
- Priority mapping from pre-extracted categories:
  Blocking = Critical (blocking issue or churn risk)
  Must-have = High (significant pain point)
  Nice-to-have = Medium (clear but non-essential request)
  Low = minor observation not in the pre-extracted feedback
- The call may be in French or English. Extract feedback in the same language as \
the transcript.
- If the transcript is unclear or too short, use the pre-extracted feedback as-is \
and note limitations in call_summary.
- Include the potential_mrr field (e.g. "480 EUR") if MRR information is found \
in the pre-extracted output header.
- Include company_size (e.g. "150 users", "50") if the number of users or employees is found \
in the pre-extracted output.
- Include company_domain (e.g. "smart-trade.net") if a domain is found \
in the pre-extracted output.

You may also receive EXISTING PRODUCT CONTEXT showing features that are already \
built, in development, or planned in Linear, as well as existing integrations from \
the product catalogue. Use this to better filter out feedback about existing \
capabilities. If the context shows a feature or integration already exists or is in \
progress, do NOT include customer requests for that feature as feedback items unless \
the customer is specifically requesting something beyond what's already described.\
"""

CALL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "call_type": {
            "type": "string",
            "enum": [
                "Demo",
                "Customer Success",
                "Discovery",
                "Support",
                "Other",
            ],
        },
        "is_external_call": {"type": "boolean"},
        "corma_participants": {"type": "array", "items": {"type": "string"}},
        "customer_participants": {"type": "array", "items": {"type": "string"}},
        "customer_company": {"type": ["string", "null"]},
        "call_summary": {"type": "string"},
        "feedback_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": [
                            "Feature Request",
                            "Bug Report",
                            "UX Issue",
                            "Integration Request",
                            "Performance Issue",
                            "Pricing Feedback",
                            "Other",
                        ],
                    },
                    "description": {"type": "string"},
                    "verbatim_quote": {"type": "string"},
                    "sentiment": {
                        "type": "string",
                        "enum": ["Positive", "Neutral", "Negative", "Frustrated"],
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["Critical", "High", "Medium", "Low"],
                    },
                    "customer_company": {"type": ["string", "null"]},
                    "customer_name": {"type": ["string", "null"]},
                },
                "required": [
                    "title",
                    "category",
                    "description",
                    "verbatim_quote",
                    "sentiment",
                    "priority",
                ],
                "additionalProperties": False,
            },
        },
        "potential_mrr": {"type": ["string", "null"]},
        "company_size": {"type": ["string", "null"]},
        "company_domain": {"type": ["string", "null"]},
        "deal_stage": {"type": ["string", "null"]},
    },
    "required": [
        "call_type",
        "is_external_call",
        "corma_participants",
        "customer_participants",
        "customer_company",
        "call_summary",
        "feedback_items",
    ],
    "additionalProperties": False,
}


CALL_TYPE_MAP = {
    "demo": "Demo",
    "product_demo": "Demo",
    "sales_demo": "Demo",
    "demonstration": "Demo",
    "customer_success": "Customer Success",
    "customer_success_call": "Customer Success",
    "customer success call": "Customer Success",
    "cs": "Customer Success",
    "cs_call": "Customer Success",
    "onboarding": "Customer Success",
    "check-in": "Customer Success",
    "checkin": "Customer Success",
    "checkup": "Customer Success",
    "follow-up": "Customer Success",
    "followup": "Customer Success",
    "follow_up": "Customer Success",
    "feedback": "Customer Success",
    "feedback_call": "Customer Success",
    "sales": "Discovery",
    "sales_call": "Discovery",
    "sales call": "Discovery",
    "prospection": "Discovery",
    "discovery": "Discovery",
    "discovery_call": "Discovery",
    "discovery call": "Discovery",
    "qualification": "Discovery",
    "negotiation": "Discovery",
    "support": "Support",
    "support_call": "Support",
    "support call": "Support",
    "internal": "Other",
    "internal_meeting": "Other",
    "internal_team_meeting": "Other",
    "internal meeting": "Other",
    "team_meeting": "Other",
    "team meeting": "Other",
    "daily": "Other",
    "standup": "Other",
    "sync": "Other",
    "synch": "Other",
    "other": "Other",
}

SENTIMENT_MAP = {
    "positive": "Positive",
    "neutral": "Neutral",
    "negative": "Negative",
    "frustrated": "Frustrated",
}

PRIORITY_MAP = {
    "critical": "Critical",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
}


CATEGORY_MAP = {
    "feature request": "Feature Request",
    "feature_request": "Feature Request",
    "feature": "Feature Request",
    "new feature": "Feature Request",
    "bug report": "Bug Report",
    "bug_report": "Bug Report",
    "bug": "Bug Report",
    "ux issue": "UX Issue",
    "ux_issue": "UX Issue",
    "ux": "UX Issue",
    "usability": "UX Issue",
    "user experience": "UX Issue",
    "integration request": "Integration Request",
    "integration_request": "Integration Request",
    "integration": "Integration Request",
    "performance issue": "Performance Issue",
    "performance_issue": "Performance Issue",
    "performance": "Performance Issue",
    "pricing feedback": "Pricing Feedback",
    "pricing_feedback": "Pricing Feedback",
    "pricing": "Pricing Feedback",
    "other": "Other",
}


def _normalize_response(data: dict) -> dict:
    """Normalize Claude's free-form JSON into the expected schema."""
    # Handle nested call_classification / call_analysis wrapper
    for wrapper_key in ("call_classification", "call_analysis", "analysis", "classification"):
        if wrapper_key in data and "call_type" not in data:
            wrapped = data.pop(wrapper_key)
            if isinstance(wrapped, dict):
                # Merge nested dict into top-level
                for k, v in wrapped.items():
                    if k not in data:
                        data[k] = v
            else:
                data["call_type"] = str(wrapped)

    # Handle nested participants wrapper
    for wrapper_key in ("participants", "speakers", "identified_speakers"):
        if wrapper_key in data and isinstance(data[wrapper_key], dict):
            participants = data.pop(wrapper_key)
            if "corma" in participants or "corma_team" in participants:
                data.setdefault("corma_participants", participants.get("corma") or participants.get("corma_team") or [])
            if "customer" in participants or "external" in participants or "customers" in participants:
                data.setdefault("customer_participants", participants.get("customer") or participants.get("external") or participants.get("customers") or [])

    # Normalize call_type to expected enum value
    raw_type = str(data.get("call_type", data.get("type", "Other"))).lower().strip().replace(" ", "_")
    # Also try with spaces for the map
    data["call_type"] = CALL_TYPE_MAP.get(raw_type, CALL_TYPE_MAP.get(raw_type.replace("_", " "), "Other"))

    # Ensure is_external_call exists — try many possible field names
    if "is_external_call" not in data:
        for alt_key in ("is_external", "external_call", "external", "is_external_meeting",
                        "has_external_participants", "involves_external"):
            if alt_key in data:
                val = data.pop(alt_key)
                data["is_external_call"] = bool(val)
                break
        else:
            # Infer from participants: if there are customer participants, it's external
            customer_parts = data.get("customer_participants", [])
            if isinstance(customer_parts, list) and len(customer_parts) > 0:
                data["is_external_call"] = True
            elif data["call_type"] == "Internal":
                data["is_external_call"] = False
            else:
                data["is_external_call"] = True  # default to external

    # Ensure required list fields exist (try alternative names)
    if "corma_participants" not in data:
        data["corma_participants"] = data.pop("corma_team", data.pop("corma_team_members", data.pop("internal_participants", [])))
    if "customer_participants" not in data:
        data["customer_participants"] = data.pop("external_participants", data.pop("prospect_participants", data.pop("customers", [])))
    if "feedback_items" not in data:
        data["feedback_items"] = data.pop("feedback", data.pop("product_feedback", data.pop("items", [])))

    # Ensure lists are actually lists
    for field in ("corma_participants", "customer_participants", "feedback_items"):
        if not isinstance(data.get(field), list):
            data[field] = []

    # Flatten participant lists: Claude sometimes returns [{"name": "X", "email": "Y"}]
    # instead of ["X"]. Convert dicts to plain name strings.
    for field in ("corma_participants", "customer_participants"):
        flattened = []
        for item in data.get(field, []):
            if isinstance(item, dict):
                flattened.append(item.get("name", item.get("full_name", str(item))))
            elif isinstance(item, str):
                flattened.append(item)
            else:
                flattened.append(str(item))
        data[field] = flattened

    # Ensure customer_company exists
    if "customer_company" not in data:
        data["customer_company"] = data.pop("company", data.pop("prospect_company", None))

    # Flatten customer_company: Claude sometimes returns a dict instead of a string
    if isinstance(data.get("customer_company"), dict):
        company_dict = data["customer_company"]
        data["customer_company"] = company_dict.get("name", company_dict.get("company_name", str(company_dict)))

    # Ensure call_summary exists
    if "call_summary" not in data:
        for alt in ("summary", "call_description", "description", "overview", "call_overview"):
            if alt in data:
                data["call_summary"] = data.pop(alt)
                break
        else:
            data["call_summary"] = "No summary available."

    # Flatten call_summary: Claude sometimes returns a dict instead of a string
    if isinstance(data.get("call_summary"), dict):
        summary_dict = data["call_summary"]
        # Try to build a readable string from the dict
        parts = []
        for k, v in summary_dict.items():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, dict):
                parts.append(str(v))
            elif isinstance(v, list):
                parts.append(", ".join(str(x) for x in v))
        data["call_summary"] = " | ".join(parts) if parts else "No summary available."

    # Normalize feedback items
    normalized_items = []
    for item in data.get("feedback_items", []):
        if not isinstance(item, dict):
            continue
        # Ensure all required fields with fallbacks
        # Claude uses many different key names for the title field.
        # Sometimes Claude puts the actual title in "feature_request" and uses "title"
        # for something else (like the call title), so check feature_request FIRST.
        if "feature_request" in item and isinstance(item["feature_request"], str) and len(item["feature_request"]) > 5:
            item["title"] = item.pop("feature_request")
        elif "feedback_text" in item and isinstance(item["feedback_text"], str) and len(item["feedback_text"]) > 5:
            item["title"] = item.pop("feedback_text")
        elif "title" not in item or not item.get("title"):
            for alt_key in ("feedback_title", "feature_request", "feedback_text",
                            "name", "feature", "feature_name", "request", "need",
                            "feedback", "summary", "item", "integration", "capability"):
                if alt_key in item and isinstance(item.get(alt_key), str) and item[alt_key]:
                    item["title"] = item.pop(alt_key)
                    break
            else:
                # Last resort: use the description if available, otherwise "Untitled feedback"
                desc = item.get("description", "")
                item["title"] = desc if desc else "Untitled feedback"
                if item["title"] == "Untitled feedback":
                    logger.warning(f"Feedback item has no title field. Keys: {list(item.keys())}")
        raw_cat = str(item.get("category", item.pop("feedback_category", item.pop("type", "Other")))).lower().strip()
        item["category"] = CATEGORY_MAP.get(raw_cat, "Other")
        item.setdefault("description", item.pop("feedback_description", item.pop("context", item.pop("detail", item.pop("details", "")))))
        item.setdefault("verbatim_quote", item.pop("quote", item.pop("customer_quote", item.pop("exact_quote", ""))))
        # Normalize sentiment
        raw_sentiment = str(item.get("sentiment", "Neutral")).lower().strip()
        item["sentiment"] = SENTIMENT_MAP.get(raw_sentiment, "Neutral")
        # Normalize priority
        raw_priority = str(item.get("priority", "Medium")).lower().strip()
        item["priority"] = PRIORITY_MAP.get(raw_priority, "Medium")
        # Remove any extra keys not in FeedbackItem
        allowed = {"title", "category", "description", "verbatim_quote", "sentiment", "priority", "customer_company", "customer_name"}
        item = {k: v for k, v in item.items() if k in allowed}
        normalized_items.append(item)
    data["feedback_items"] = normalized_items

    # Ensure potential_mrr exists
    if "potential_mrr" not in data:
        data["potential_mrr"] = data.pop("mrr", data.pop("potential_revenue",
                                  data.pop("estimated_mrr", data.pop("mrr_potential", None))))

    # Ensure company_size exists
    if "company_size" not in data:
        data["company_size"] = data.pop("size", data.pop("number_of_users",
                                  data.pop("users", data.pop("employee_count", None))))

    # Ensure company_domain exists
    if "company_domain" not in data:
        data["company_domain"] = data.pop("domain", data.pop("customer_domain",
                                    data.pop("website", None)))

    # Ensure deal_stage exists
    if "deal_stage" not in data:
        data["deal_stage"] = data.pop("stage", data.pop("deal_status", None))

    # Remove unexpected top-level keys to avoid Pydantic errors
    allowed_top = {"call_type", "is_external_call", "corma_participants", "customer_participants",
                   "customer_company", "call_summary", "feedback_items", "potential_mrr",
                   "company_size", "company_domain", "deal_stage"}
    data = {k: v for k, v in data.items() if k in allowed_top}

    return data


class TranscriptAnalyzer:
    def __init__(self) -> None:
        self.client = anthropic.Anthropic()

    def analyze_call(
        self, call: CallMetadata, product_context: str | None = None
    ) -> CallAnalysisResult | None:
        """Analyze a call transcript and return structured feedback."""
        if not call.transcript_text:
            return None

        if len(call.transcript_text) > MAX_TRANSCRIPT_CHARS:
            return self._chunk_and_analyze(call, product_context)

        return self._analyze_single(call, call.transcript_text, product_context)

    def _analyze_single(
        self,
        call: CallMetadata,
        transcript_text: str,
        product_context: str | None = None,
    ) -> CallAnalysisResult | None:
        user_message = self._build_user_message(
            call, transcript_text, product_context
        )

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=8192,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

        except anthropic.APIError as e:
            logger.error(f"Claude API error for call {call.uuid}: {e}")
            return None

        if response.stop_reason == "refusal":
            logger.warning(f"Claude refused to analyze call {call.uuid}")
            return None

        # Extract JSON from response content
        text_content = ""
        for block in response.content:
            if block.type == "text":
                text_content = block.text
                break

        if not text_content:
            logger.error(f"No text content in Claude response for call {call.uuid}")
            return None

        # Strip markdown code fences if present
        cleaned = text_content.strip()
        if cleaned.startswith("```"):
            # Remove opening fence (```json or ```)
            first_newline = cleaned.index("\n")
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try removing control characters (keep \n, \r, \t which are valid JSON whitespace)
            cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', cleaned)
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                # Try extracting only the first JSON object (Claude sometimes appends text)
                try:
                    decoder = json.JSONDecoder()
                    parsed, _ = decoder.raw_decode(cleaned)
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Failed to parse Claude response for call {call.uuid}: {e}\n"
                        f"Response preview: {cleaned[:500]}"
                    )
                    return None

        try:
            normalized = _normalize_response(parsed)
            return CallAnalysisResult(**normalized)
        except (ValueError, TypeError) as e:
            logger.error(
                f"Failed to validate Claude response for call {call.uuid}: {e}\n"
                f"Normalized data preview: {str(normalized)[:500]}"
            )
            return None

    def _chunk_and_analyze(
        self, call: CallMetadata, product_context: str | None = None
    ) -> CallAnalysisResult | None:
        """Split a long transcript into overlapping chunks and merge results."""
        transcript = call.transcript_text or ""
        chunk_size = 150_000
        overlap = 5_000

        chunks: list[str] = []
        start = 0
        while start < len(transcript):
            end = start + chunk_size
            chunks.append(transcript[start:end])
            start = end - overlap

        logger.info(
            f"Splitting transcript for call {call.uuid} into {len(chunks)} chunks"
        )

        all_feedback: list[FeedbackItem] = []
        first_result: CallAnalysisResult | None = None

        for i, chunk in enumerate(chunks):
            logger.debug(f"Analyzing chunk {i + 1}/{len(chunks)} for call {call.uuid}")
            result = self._analyze_single(call, chunk, product_context)
            if result is None:
                continue
            if first_result is None:
                first_result = result
            all_feedback.extend(result.feedback_items)

        if first_result is None:
            return None

        # Deduplicate by title similarity
        seen_titles: set[str] = set()
        unique_feedback: list[FeedbackItem] = []
        for item in all_feedback:
            key = item.title.lower().strip()
            if key not in seen_titles:
                seen_titles.add(key)
                unique_feedback.append(item)

        first_result.feedback_items = unique_feedback
        return first_result

    def pick_best_match(
        self,
        feedback_title: str,
        feedback_description: str,
        candidates: list[dict],
    ) -> dict | None:
        """Use Claude to pick the best matching Linear issue/project from candidates.

        Returns the best match dict, or None if Claude says NONE match.
        Returns None on failure (safer to create a new ticket than match wrongly).
        """
        if not candidates:
            return None

        # Build a concise prompt for Claude
        candidate_lines = []
        for i, c in enumerate(candidates[:10], 1):
            identifier = c.get("identifier", c.get("id", "?"))
            name = c.get("title", c.get("name", "Unknown"))
            state = c.get("state", {}).get("name", c.get("state", ""))
            candidate_lines.append(f"{i}. {identifier}: {name} ({state})")

        candidates_text = "\n".join(candidate_lines)

        prompt = (
            f"Given this product feedback:\n"
            f"Title: {feedback_title}\n"
            f"Description: {feedback_description}\n\n"
            f"Which of these Linear items is the best match? "
            f"The match must be about the SAME specific product, tool, or feature. "
            f"Reply with ONLY the number (e.g. '1') or 'NONE' if none match.\n"
            f"IMPORTANT: Different products/tools are NOT a match even if they "
            f"are in the same category (e.g. RAMP ≠ BREX, Slack ≠ Teams).\n\n"
            f"{candidates_text}"
        )

        try:
            response = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=32,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            if text.upper() == "NONE":
                return None
            # Parse the number
            match = re.search(r"\d+", text)
            if match:
                idx = int(match.group()) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            logger.warning("pick_best_match failed, returning None (no match)")

        return None

    @staticmethod
    def _build_user_message(
        call: CallMetadata,
        transcript_text: str,
        product_context: str | None = None,
    ) -> str:
        speaker_lines: list[str] = []
        for s in call.speakers:
            label = "CORMA TEAM" if s.is_corma_team else "EXTERNAL"
            email_part = f" ({s.email})" if s.email else ""
            speaker_lines.append(f"- {s.name}{email_part} — {label}")

        speakers_block = "\n".join(speaker_lines) if speaker_lines else "- Unknown"

        duration_str = ""
        if call.duration:
            minutes = int(call.duration // 60)
            seconds = int(call.duration % 60)
            duration_str = f"{minutes}m {seconds}s"

        # Build prompt output section
        prompt_section = ""
        if call.feedback_prompt_output:
            prompt_section = f"""
PRE-EXTRACTED FEEDBACK (from Leexi AI):
{call.feedback_prompt_output}

"""

        # Build product context section
        context_section = ""
        if product_context:
            context_section = f"""
EXISTING PRODUCT CONTEXT (from internal systems):
{product_context}

IMPORTANT: Use the above context to filter out feedback about features or \
integrations that already exist or are already being built. Only include feedback \
about capabilities that are NOT listed above, unless the customer is requesting \
something specifically beyond what's described.

"""

        return f"""\
Analyze the following sales call data from Corma.
{context_section}\
CALL METADATA:
- Title: {call.title or "Untitled"}
- Date: {call.performed_at or "Unknown"}
- Duration: {duration_str or "Unknown"}
- Direction: {call.direction or "Unknown"}

IDENTIFIED SPEAKERS:
{speakers_block}

KNOWN CORMA TEAM MEMBERS: Heloise, Nikolai, Alessandro, Jean, Solal, Yann, \
Quentin, Louis, Samuel
{prompt_section}\
FULL TRANSCRIPT:
{transcript_text}

Analyze this call. For each pre-extracted feedback point, check whether the Corma \
rep already explained or demonstrated that capability during the call — if so, \
EXCLUDE it. Also cross-reference with the EXISTING PRODUCT CONTEXT above to \
exclude feedback about features or integrations that already exist or are in \
development. Only include feedback about features Corma does NOT currently have. \
Be very specific: each feedback item should be a concrete feature or integration \
that a product manager could turn into a ticket. Prefer an empty feedback_items \
list over vague or already-existing features. If MRR information is available in \
the pre-extracted header, include it in potential_mrr. Respond with valid JSON \
matching the required schema."""
