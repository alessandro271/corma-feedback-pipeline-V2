import logging
import time
from collections import Counter

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import NOTION_API_VERSION, NOTION_BASE_URL, NOTION_MIN_REQUEST_INTERVAL
from models import CallAnalysisResult, CallMetadata, FeedbackItem

logger = logging.getLogger("corma-feedback.notion")


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


class NotionClient:
    def __init__(
        self,
        api_key: str,
        database_id: str,
        weekly_parent_page_id: str,
    ) -> None:
        self.database_id = database_id
        self.weekly_parent_page_id = weekly_parent_page_id
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Notion-Version": NOTION_API_VERSION,
                "Content-Type": "application/json",
            }
        )
        self._last_request_time: float = 0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < NOTION_MIN_REQUEST_INTERVAL:
            time.sleep(NOTION_MIN_REQUEST_INTERVAL - elapsed)

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        reraise=True,
    )
    def _request(
        self,
        method: str,
        endpoint: str,
        json_body: dict | None = None,
    ) -> dict:
        self._throttle()
        url = f"{NOTION_BASE_URL}/{endpoint.lstrip('/')}"
        logger.debug(f"{method} {url}")
        resp = self.session.request(method, url, json=json_body, timeout=30)
        self._last_request_time = time.time()
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", "1"))
            logger.warning(f"Notion rate limited, retrying after {retry_after}s")
            time.sleep(retry_after)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Database queries
    # ------------------------------------------------------------------

    def query_existing_entries(self, week_label: str) -> set[str]:
        """Return titles of feedback entries already created for this week."""
        titles: set[str] = set()
        start_cursor: str | None = None

        while True:
            body: dict = {
                "filter": {
                    "property": "Week",
                    "rich_text": {"equals": week_label},
                }
            }
            if start_cursor:
                body["start_cursor"] = start_cursor

            data = self._request(
                "POST", f"databases/{self.database_id}/query", json_body=body
            )
            for page in data.get("results", []):
                title_prop = page.get("properties", {}).get("Feedback Title", {})
                title_items = title_prop.get("title", [])
                if title_items:
                    titles.add(title_items[0].get("plain_text", ""))

            if data.get("has_more") and data.get("next_cursor"):
                start_cursor = data["next_cursor"]
            else:
                break

        logger.info(f"Found {len(titles)} existing entries for week {week_label}")
        return titles

    def query_entries_for_date_range(
        self, from_date: str, to_date: str
    ) -> list[dict]:
        """Query all feedback entries within a date range.

        Returns raw Notion page objects for aggregation (e.g. weekly digest).
        """
        all_pages: list[dict] = []
        start_cursor: str | None = None

        while True:
            body: dict = {
                "filter": {
                    "and": [
                        {
                            "property": "Date",
                            "date": {"on_or_after": from_date},
                        },
                        {
                            "property": "Date",
                            "date": {"on_or_before": to_date},
                        },
                    ]
                },
                "sorts": [{"property": "Date", "direction": "descending"}],
            }
            if start_cursor:
                body["start_cursor"] = start_cursor

            data = self._request(
                "POST", f"databases/{self.database_id}/query", json_body=body
            )
            all_pages.extend(data.get("results", []))

            if data.get("has_more") and data.get("next_cursor"):
                start_cursor = data["next_cursor"]
            else:
                break

        logger.info(f"Found {len(all_pages)} entries for {from_date} to {to_date}")
        return all_pages

    def clear_all_entries(self) -> int:
        """Archive (soft-delete) all entries in the database. Returns count deleted."""
        deleted = 0
        start_cursor: str | None = None

        while True:
            body: dict = {"page_size": 100}
            if start_cursor:
                body["start_cursor"] = start_cursor

            data = self._request(
                "POST", f"databases/{self.database_id}/query", json_body=body
            )
            pages = data.get("results", [])
            if not pages:
                break

            for page in pages:
                page_id = page["id"]
                self._request(
                    "PATCH", f"pages/{page_id}", json_body={"archived": True}
                )
                deleted += 1

            if data.get("has_more") and data.get("next_cursor"):
                start_cursor = data["next_cursor"]
            else:
                break

        logger.info(f"Archived {deleted} entries from database")
        return deleted

    def update_database_schema(self) -> None:
        """Update database schema with correct select options.

        Safe to run multiple times — Notion will merge/update properties.
        """
        body = {
            "properties": {
                "Status": {
                    "select": {
                        "options": [
                            {"name": "New", "color": "blue"},
                            {"name": "Added to Linear", "color": "purple"},
                            {"name": "In Progress", "color": "yellow"},
                            {"name": "Shipped", "color": "green"},
                            {"name": "Won't Do", "color": "gray"},
                        ]
                    }
                },
                "MRR Potential": {"rich_text": {}},
                "Call Type": {
                    "select": {
                        "options": [
                            {"name": "Demo", "color": "blue"},
                            {"name": "Customer Success", "color": "green"},
                            {"name": "Discovery", "color": "purple"},
                            {"name": "Support", "color": "orange"},
                            {"name": "Other", "color": "gray"},
                        ]
                    }
                },
            }
        }
        self._request("PATCH", f"databases/{self.database_id}", json_body=body)
        logger.info("Database schema updated")

    # ------------------------------------------------------------------
    # Create feedback entry
    # ------------------------------------------------------------------

    def create_feedback_entry(
        self,
        item: FeedbackItem,
        call: CallMetadata,
        analysis: CallAnalysisResult,
        week_label: str,
    ) -> str:
        """Create a single feedback page in the Notion database. Returns page ID."""

        def _text(content: str) -> list[dict]:
            return [{"type": "text", "text": {"content": content[:2000]}}]

        # Determine call type: prefer Leexi stage, fall back to Claude's inference
        call_type = call.leexi_call_type or analysis.call_type or "Other"

        # Determine Corma contact: prefer Leexi owner, fall back to Claude's participants
        corma_contact = (
            call.leexi_owner_name
            or (", ".join(analysis.corma_participants) if analysis.corma_participants else "")
        )

        properties = {
            "Feedback Title": {"title": _text(item.title)},
            "Category": {"select": {"name": item.category}},
            "Description": {"rich_text": _text(item.description)},
            "Verbatim Quote": {"rich_text": _text(item.verbatim_quote)},
            "Customer / Company": {
                "rich_text": _text(
                    item.customer_company
                    or analysis.customer_company
                    or call.leexi_company_name
                    or call.leexi_contact_email
                    or "Unknown"
                )
            },
            "Call Type": {"select": {"name": call_type}},
            "Corma Contact": {
                "rich_text": _text(corma_contact or "Unknown")
            },
            "Priority": {"select": {"name": item.priority}},
            "Sentiment": {"select": {"name": item.sentiment}},
            "Week": {"rich_text": _text(week_label)},
        }

        if call.performed_at:
            properties["Date"] = {"date": {"start": call.performed_at[:10]}}

        if call.leexi_url:
            properties["Source Call URL"] = {"url": call.leexi_url}

        # New v2 properties
        properties["Status"] = {"select": {"name": "New"}}
        if analysis.potential_mrr:
            properties["MRR Potential"] = {"rich_text": _text(analysis.potential_mrr)}

        body = {
            "parent": {"database_id": self.database_id},
            "properties": properties,
        }

        data = self._request("POST", "pages", json_body=body)
        page_id = data.get("id", "")
        logger.info(f"Created feedback entry: {item.title!r} (page {page_id})")
        return page_id

    # ------------------------------------------------------------------
    # Weekly summary page
    # ------------------------------------------------------------------

    def create_weekly_summary_page(
        self,
        week_label: str,
        results: list[tuple[CallMetadata, CallAnalysisResult]],
        stats: dict,
    ) -> str:
        """Create a rich summary page under the weekly parent page."""

        # Create the page first
        page_body = {
            "parent": {"page_id": self.weekly_parent_page_id},
            "properties": {
                "title": [
                    {
                        "type": "text",
                        "text": {
                            "content": f"Product Feedback — Week {week_label}"
                        },
                    }
                ]
            },
        }
        page_data = self._request("POST", "pages", json_body=page_body)
        page_id = page_data["id"]
        logger.info(f"Created weekly summary page: {page_id}")

        # Build content blocks
        blocks = self._build_summary_blocks(week_label, results, stats)

        # Append in batches of 100
        for i in range(0, len(blocks), 100):
            batch = blocks[i : i + 100]
            self._request(
                "PATCH",
                f"blocks/{page_id}/children",
                json_body={"children": batch},
            )

        return page_id

    def _build_summary_blocks(
        self,
        week_label: str,
        results: list[tuple[CallMetadata, CallAnalysisResult]],
        stats: dict,
    ) -> list[dict]:
        blocks: list[dict] = []

        # --- Overview ---
        blocks.append(_heading2("Overview"))
        blocks.append(
            _paragraph(
                f"Total calls fetched: {stats.get('total_calls', 0)}\n"
                f"Calls analyzed: {stats.get('analyzed', 0)}\n"
                f"Skipped (empty transcript): {stats.get('skipped_empty', 0)}\n"
                f"Skipped (internal): {stats.get('skipped_internal', 0)}\n"
                f"Feedback items created: {stats.get('feedback_items_created', 0)}\n"
                f"Errors: {stats.get('errors', 0)}"
            )
        )
        blocks.append(_divider())

        # Collect all feedback items
        all_feedback: list[tuple[FeedbackItem, CallMetadata, CallAnalysisResult]] = []
        for call, analysis in results:
            for item in analysis.feedback_items:
                all_feedback.append((item, call, analysis))

        # --- By Category ---
        blocks.append(_heading2("Feedback by Category"))
        cat_counts = Counter(item.category for item, _, _ in all_feedback)
        for cat, count in cat_counts.most_common():
            blocks.append(_bullet(f"{cat}: {count}"))

        blocks.append(_divider())

        # --- By Priority ---
        blocks.append(_heading2("Feedback by Priority"))
        pri_counts = Counter(item.priority for item, _, _ in all_feedback)
        for pri in ["Critical", "High", "Medium", "Low"]:
            if pri in pri_counts:
                blocks.append(_bullet(f"{pri}: {pri_counts[pri]}"))

        blocks.append(_divider())

        # --- Top Priority Items ---
        high_priority = [
            (item, call, analysis)
            for item, call, analysis in all_feedback
            if item.priority in ("Critical", "High")
        ]
        if high_priority:
            blocks.append(_heading2("Top Priority Items"))
            for item, call, analysis in high_priority:
                company = item.customer_company or item.customer_name or "Unknown"
                contact = ", ".join(analysis.corma_participants) or "Unknown"
                blocks.append(
                    _paragraph(
                        f"▸ {item.title} ({item.category}, {item.priority})\n"
                        f"  Customer: {company} | Contact: {contact}\n"
                        f'  "{item.verbatim_quote[:500]}"\n'
                        f"  {item.description}"
                    )
                )
            blocks.append(_divider())

        # --- All Calls ---
        blocks.append(_heading2("All Calls Analyzed"))
        for call, analysis in results:
            feedback_count = len(analysis.feedback_items)
            company = analysis.customer_company or "—"
            contact = ", ".join(analysis.corma_participants) or "—"
            title = call.title or "Untitled"
            blocks.append(
                _bullet(
                    f"{title} | {analysis.call_type} | {company} "
                    f"| {contact} | {feedback_count} feedback item(s)"
                )
            )

        return blocks


# ------------------------------------------------------------------
# Block helpers
# ------------------------------------------------------------------


def _rich_text(content: str) -> list[dict]:
    # Split into chunks of 2000 chars if needed
    chunks = []
    for i in range(0, len(content), 2000):
        chunks.append({"type": "text", "text": {"content": content[i : i + 2000]}})
    return chunks


def _heading2(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": _rich_text(text)},
    }


def _paragraph(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text)},
    }


def _bullet(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text(text)},
    }


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}
