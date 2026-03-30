"""Slack notification client using Incoming Webhooks and Web API.

Supports both webhook-based messaging (legacy) and Slack Web API
(chat.postMessage) for thread replies after Linear sync.
"""

import logging
from typing import Any

import requests

logger = logging.getLogger("corma-feedback.slack")

SLACK_API_URL = "https://slack.com/api"

# Emoji mapping for priority levels
_PRIORITY_EMOJI = {
    "Critical": ":red_circle:",
    "High": ":large_orange_circle:",
    "Medium": ":large_yellow_circle:",
    "Low": ":white_circle:",
}


class SlackClient:
    """Send formatted messages to Slack channels via webhooks or Web API."""

    def __init__(
        self,
        webhook_url: str,
        weekly_webhook_url: str | None = None,
        bot_token: str | None = None,
        channel_id: str | None = None,
    ) -> None:
        self.webhook_url = webhook_url
        self.weekly_webhook_url = weekly_webhook_url or webhook_url
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.session = requests.Session()

    @property
    def _use_web_api(self) -> bool:
        """Whether we can use the Slack Web API (bot token + channel configured)."""
        return bool(self.bot_token and self.channel_id)

    # ------------------------------------------------------------------
    # Low-level posting methods
    # ------------------------------------------------------------------

    def _post_webhook(self, url: str, payload: dict[str, Any]) -> bool:
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

    def _post_web_api(
        self,
        blocks: list[dict],
        text: str,
        thread_ts: str | None = None,
    ) -> str | None:
        """Post a message using Slack Web API. Returns message ts on success."""
        if not self.bot_token or not self.channel_id:
            return None

        payload: dict[str, Any] = {
            "channel": self.channel_id,
            "text": text,
            "blocks": blocks,
        }
        if thread_ts:
            payload["thread_ts"] = thread_ts

        try:
            resp = self.session.post(
                f"{SLACK_API_URL}/chat.postMessage",
                json=payload,
                headers={"Authorization": f"Bearer {self.bot_token}"},
                timeout=10,
            )
            data = resp.json()
            if not data.get("ok"):
                logger.error(f"Slack Web API error: {data.get('error', 'unknown')}")
                return None
            ts = data.get("ts")
            logger.info(f"Slack message posted (ts={ts})")
            return ts
        except requests.RequestException as e:
            logger.error(f"Slack Web API request failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Per-call feedback message
    # ------------------------------------------------------------------

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
        company_size: str | None = None,
        company_domain: str | None = None,
    ) -> str | None:
        """Send a per-call feedback notification to Slack.

        Returns the message timestamp (ts) if using Web API, or None if using
        webhooks (webhooks don't return a ts).
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
                ]
                + ([{"type": "mrkdwn", "text": f"*Size:* {company_size}"}] if company_size else [])
                + ([{"type": "mrkdwn", "text": f"*Domain:* {company_domain}"}] if company_domain else []),
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

        fallback_text = f"New product feedback from {company_display} ({len(feedback_items)} items)"

        # Prefer Web API (returns ts for threading), fall back to webhook
        if self._use_web_api:
            return self._post_web_api(blocks, fallback_text)
        else:
            self._post_webhook(self.webhook_url, {"text": fallback_text, "blocks": blocks})
            return None  # Webhook doesn't return ts

    # ------------------------------------------------------------------
    # Linear sync thread reply
    # ------------------------------------------------------------------

    def send_linear_sync_reply(
        self,
        thread_ts: str,
        sync_results: list[dict],
        customer_summary: str | None = None,
    ) -> bool:
        """Reply to a Slack thread with the Linear sync summary.

        Args:
            thread_ts: The timestamp of the original per-call message.
            sync_results: List of dicts with keys: feedback_title, action,
                          issue_identifier, issue_url.
            customer_summary: Optional text about customer card actions.
        """
        if not self._use_web_api:
            logger.warning("Cannot send thread reply: Slack Web API not configured")
            return False

        lines = [":white_check_mark: *Linear sync complete:*"]
        for result in sync_results:
            title = result.get("feedback_title", "?")
            action = result.get("action", "?")
            identifier = result.get("issue_identifier", "")
            url = result.get("issue_url", "")

            if action == "created":
                issue_ref = f"<{url}|{identifier}>" if url else identifier
                lines.append(f"• \"{title}\" → Created {issue_ref} (Backlog)")
            elif action == "customer_need_added":
                issue_ref = f"<{url}|{identifier}>" if url else identifier
                lines.append(
                    f"• \"{title}\" → Added as customer request on {issue_ref}"
                )
            elif action == "error":
                error_msg = result.get("error_message", "unknown error")
                lines.append(f"• \"{title}\" → :warning: Error: {error_msg}")

        if customer_summary:
            lines.append("")
            lines.append(f":bust_in_silhouette: {customer_summary}")

        text = "\n".join(lines)

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": text},
            }
        ]

        ts = self._post_web_api(blocks, text, thread_ts=thread_ts)
        return ts is not None

    # ------------------------------------------------------------------
    # Weekly digest (unchanged — still uses webhook)
    # ------------------------------------------------------------------

    def send_weekly_digest(
        self,
        week_label: str,
        total_calls_with_feedback: int,
        total_feedback_items: int,
        top_features: list[dict | str],
        mrr_details: list[dict],
        total_mrr: str | None,
    ) -> bool:
        """Send the weekly digest summary to Slack."""
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
        return self._post_webhook(self.weekly_webhook_url, {"text": text, "blocks": blocks})
