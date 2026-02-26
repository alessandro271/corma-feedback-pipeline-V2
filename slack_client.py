"""Slack notification client using Incoming Webhooks."""

import logging
from typing import Any

import requests

logger = logging.getLogger("corma-feedback.slack")

# Emoji mapping for priority levels
_PRIORITY_EMOJI = {
    "Critical": ":red_circle:",
    "High": ":large_orange_circle:",
    "Medium": ":large_yellow_circle:",
    "Low": ":white_circle:",
}


class SlackClient:
    """Send formatted messages to Slack channels via incoming webhooks."""

    def __init__(
        self, webhook_url: str, weekly_webhook_url: str | None = None
    ) -> None:
        self.webhook_url = webhook_url
        self.weekly_webhook_url = weekly_webhook_url or webhook_url
        self.session = requests.Session()

    def _post(self, url: str, payload: dict[str, Any]) -> bool:
        """Post a message payload to a Slack webhook. Returns True on success."""
        if not url:
            logger.warning("Slack webhook URL not configured, skipping notification")
            return False
        try:
            resp = self.session.post(url, json=payload, timeout=10)
            if resp.status_code != 200 or resp.text != "ok":
                logger.error(f"Slack webhook error: {resp.status_code} {resp.text}")
                return False
            logger.info("Slack notification sent successfully")
            return True
        except requests.RequestException as e:
            logger.error(f"Slack webhook request failed: {e}")
            return False

    def send_call_feedback(
        self,
        call_title: str,
        call_date: str,
        call_type: str,
        customer_company: str | None,
        corma_contacts: list[str],
        feedback_items: list[dict],
        potential_mrr: str | None,
        leexi_url: str | None,
    ) -> bool:
        """Send a per-call feedback notification to Slack.

        Args:
            call_title: Title of the call.
            call_date: Date string (YYYY-MM-DD).
            call_type: Type of call (Demo, Sales, etc.).
            customer_company: Company name if known.
            corma_contacts: Corma team members on the call.
            feedback_items: List of dicts with keys: title, category, priority, description.
            potential_mrr: MRR potential string (e.g. "480 EUR") if available.
            leexi_url: Direct link to the call in Leexi.
        """
        company_display = customer_company or "Unknown company"
        contacts_display = ", ".join(corma_contacts) if corma_contacts else "Unknown"

        # Header
        blocks: list[dict] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"New Product Feedback from {company_display}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Call:* {call_title}"},
                    {"type": "mrkdwn", "text": f"*Date:* {call_date}"},
                    {"type": "mrkdwn", "text": f"*Type:* {call_type}"},
                    {"type": "mrkdwn", "text": f"*Corma Contact:* {contacts_display}"},
                ],
            },
        ]

        if potential_mrr:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f":moneybag: *MRR Potential:* {potential_mrr}",
                    },
                }
            )

        # Feedback items
        for item in feedback_items:
            priority = item.get("priority", "Medium")
            emoji = _PRIORITY_EMOJI.get(priority, ":white_circle:")

            blocks.append({"type": "divider"})
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{emoji} *{item['title']}* ({item['category']})\n"
                            f"{item['description']}"
                        ),
                    },
                }
            )

        # Link to Leexi
        if leexi_url:
            blocks.append({"type": "divider"})
            blocks.append(
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f":link: <{leexi_url}|View call in Leexi>",
                        }
                    ],
                }
            )

        # Fallback text for notifications
        text = f"New product feedback from {company_display} ({len(feedback_items)} items)"
        return self._post(self.webhook_url, {"text": text, "blocks": blocks})

    def send_weekly_digest(
        self,
        week_label: str,
        total_calls_with_feedback: int,
        total_feedback_items: int,
        top_features: list[dict | str],
        mrr_details: list[dict],
        total_mrr: str | None,
    ) -> bool:
        """Send the weekly digest summary to Slack.

        Args:
            week_label: Week identifier (e.g. "2026-W08").
            total_calls_with_feedback: Number of calls where feedback was found.
            total_feedback_items: Total number of feedback items created.
            top_features: List of dicts with keys: title, company, mrr (or plain strings).
            mrr_details: List of dicts with keys: company, mrr (deduped by company).
            total_mrr: Aggregated MRR string if computable.
        """
        blocks: list[dict] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Weekly Product Feedback Digest — {week_label}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*:phone: Calls with feedback:* {total_calls_with_feedback}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*:memo: Total feedback items:* {total_feedback_items}",
                    },
                ],
            },
        ]

        if top_features:
            lines: list[str] = []
            for f in top_features[:10]:
                if isinstance(f, dict):
                    title = f.get("title", "")
                    company = f.get("company", "")
                    mrr = f.get("mrr", "")
                    # Build parenthetical: (Company — MRR) or (Company) or just title
                    parts = []
                    if company and company != "Unknown":
                        parts.append(company)
                    if mrr:
                        parts.append(mrr)
                    suffix = f" ({' — '.join(parts)})" if parts else ""
                    lines.append(f"• {title}{suffix}")
                else:
                    lines.append(f"• {f}")
            feature_list = "\n".join(lines)
            blocks.append({"type": "divider"})
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*:star: Top features/requests:*\n{feature_list}",
                    },
                }
            )

        if mrr_details:
            mrr_lines = "\n".join(
                f"• {d['company']}: {d['mrr']}" for d in mrr_details
            )
            total_line = f"\n*Total MRR Potential:* {total_mrr}" if total_mrr else ""
            blocks.append({"type": "divider"})
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*:moneybag: MRR Details:*\n{mrr_lines}{total_line}",
                    },
                }
            )

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": ":robot_face: Generated by Corma Feedback Pipeline",
                    }
                ],
            }
        )

        text = f"Weekly feedback digest — {week_label}: {total_calls_with_feedback} calls, {total_feedback_items} feedback items"
        return self._post(self.weekly_webhook_url, {"text": text, "blocks": blocks})
