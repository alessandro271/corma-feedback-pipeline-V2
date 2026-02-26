#!/usr/bin/env python3
"""Corma Customer Feedback Pipeline v2.

Fetches calls from Leexi, filters by the "product feedback prompt",
analyzes with Claude (prompt output + transcript), pushes to Notion,
and sends Slack notifications.

Modes:
  - Default (daily): Process recent calls, create Notion entries, send Slack per-call.
  - --weekly-digest: Query Notion for past 7 days and send a summary to Slack.
  - --update-schema: Add new properties (Status, MRR) to an existing Notion database.
"""

import argparse
import json
import sys
from datetime import date, timedelta

from config import (
    DEFAULT_LOOKBACK_DAYS,
    LEEXI_API_KEY_ID,
    LEEXI_API_KEY_SECRET,
    NOTION_API_KEY,
    NOTION_DATABASE_ID,
    NOTION_WEEKLY_PARENT_PAGE_ID,
    SLACK_WEBHOOK_URL,
    SLACK_WEEKLY_WEBHOOK_URL,
    setup_logging,
)
from analyzer import TranscriptAnalyzer
from leexi_client import LeexiClient
from models import CallAnalysisResult, CallMetadata
from notion_client import NotionClient
from slack_client import SlackClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corma Customer Feedback Pipeline"
    )
    parser.add_argument(
        "--from-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to yesterday for daily mode.",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze calls but do not write to Notion or send Slack.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--weekly-digest",
        action="store_true",
        help="Generate weekly Slack digest from Notion data (no call processing).",
    )
    parser.add_argument(
        "--update-schema",
        action="store_true",
        help="Add new properties (Status, MRR) to existing Notion database.",
    )
    parser.add_argument(
        "--clear-database",
        action="store_true",
        help="Archive all existing entries in the Notion database.",
    )
    return parser.parse_args()


def validate_config(mode: str = "daily") -> list[str]:
    """Return a list of missing config values for the given mode."""
    missing = []

    if mode == "daily":
        if not LEEXI_API_KEY_ID:
            missing.append("LEEXI_API_KEY_ID")
        if not LEEXI_API_KEY_SECRET:
            missing.append("LEEXI_API_KEY_SECRET")
        if not NOTION_API_KEY:
            missing.append("NOTION_API_KEY")
        if not NOTION_DATABASE_ID:
            missing.append("NOTION_DATABASE_ID")
        # ANTHROPIC_API_KEY is read by the anthropic SDK from env directly
    elif mode in ("weekly-digest", "update-schema"):
        if not NOTION_API_KEY:
            missing.append("NOTION_API_KEY")
        if not NOTION_DATABASE_ID:
            missing.append("NOTION_DATABASE_ID")

    return missing


def main() -> None:
    args = parse_args()
    logger = setup_logging(verbose=args.verbose)

    # --- Schema update mode ---
    if args.update_schema:
        missing = validate_config("update-schema")
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            sys.exit(1)
        notion = NotionClient(NOTION_API_KEY, NOTION_DATABASE_ID, NOTION_WEEKLY_PARENT_PAGE_ID)
        notion.update_database_schema()
        logger.info("Schema updated successfully")
        return

    # --- Clear database mode ---
    if args.clear_database:
        missing = validate_config("update-schema")
        if missing:
            logger.error(f"Missing required environment variables: {', '.join(missing)}")
            sys.exit(1)
        notion = NotionClient(NOTION_API_KEY, NOTION_DATABASE_ID, NOTION_WEEKLY_PARENT_PAGE_ID)
        count = notion.clear_all_entries()
        logger.info(f"Cleared {count} entries from database")
        return

    # --- Weekly digest mode ---
    if args.weekly_digest:
        run_weekly_digest(args, logger)
        return

    # --- Normal daily pipeline ---
    run_daily_pipeline(args, logger)


def run_weekly_digest(args, logger) -> None:
    """Generate weekly Slack digest from Notion database entries."""
    missing = validate_config("weekly-digest")
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)

    today = date.today()
    from_date = args.from_date or (today - timedelta(days=7)).isoformat()
    to_date = args.to_date or today.isoformat()

    end_date = date.fromisoformat(to_date)
    iso_cal = (end_date - timedelta(days=1)).isocalendar()
    week_label = f"{iso_cal[0]}-W{iso_cal[1]:02d}"

    logger.info("=== Weekly Digest Mode ===")
    logger.info(f"Date range: {from_date} to {to_date}")
    logger.info(f"Week label: {week_label}")

    notion = NotionClient(NOTION_API_KEY, NOTION_DATABASE_ID, NOTION_WEEKLY_PARENT_PAGE_ID)
    slack = (
        SlackClient(SLACK_WEBHOOK_URL, SLACK_WEEKLY_WEBHOOK_URL or None)
        if SLACK_WEBHOOK_URL and not args.dry_run
        else None
    )

    # Query Notion for this week's entries
    pages = notion.query_entries_for_date_range(from_date, to_date)

    if not pages:
        logger.info("No feedback entries found for this period")
        if slack:
            slack.send_weekly_digest(
                week_label=week_label,
                total_calls_with_feedback=0,
                total_feedback_items=0,
                top_features=[],
                mrr_details=[],
                total_mrr=None,
            )
        return

    # Aggregate data from Notion pages
    feedback_titles: list[str] = []
    mrr_details: list[dict] = []
    call_urls: set[str] = set()

    for page in pages:
        props = page.get("properties", {})

        # Feedback title
        title_prop = props.get("Feedback Title", {}).get("title", [])
        if title_prop:
            feedback_titles.append(title_prop[0].get("plain_text", ""))

        # Company
        company_prop = props.get("Customer / Company", {}).get("rich_text", [])
        company = company_prop[0].get("plain_text", "") if company_prop else ""

        # MRR
        mrr_prop = props.get("MRR Potential", {}).get("rich_text", [])
        mrr = mrr_prop[0].get("plain_text", "") if mrr_prop else ""
        if mrr:
            mrr_details.append({"company": company or "Unknown", "mrr": mrr})

        # Count unique calls
        url_prop = props.get("Source Call URL", {}).get("url")
        if url_prop:
            call_urls.add(url_prop)

    logger.info(
        f"Digest: {len(pages)} feedback items from {len(call_urls)} calls, "
        f"{len(mrr_details)} with MRR data"
    )

    if slack:
        slack.send_weekly_digest(
            week_label=week_label,
            total_calls_with_feedback=len(call_urls),
            total_feedback_items=len(pages),
            top_features=feedback_titles[:10],
            mrr_details=mrr_details,
            total_mrr=None,
        )
        logger.info("Weekly Slack digest sent")
    else:
        logger.info(
            f"DRY RUN: Would send digest with {len(pages)} items "
            f"from {len(call_urls)} calls"
        )


def run_daily_pipeline(args, logger) -> None:
    """Main daily pipeline: fetch calls, filter by prompt, analyze, store, notify."""
    missing = validate_config("daily")
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        if not args.dry_run or "LEEXI_API_KEY_ID" in missing:
            sys.exit(1)

    today = date.today()
    from_date = args.from_date or (today - timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat()
    to_date = args.to_date or today.isoformat()

    end_date = date.fromisoformat(to_date)
    iso_cal = (end_date - timedelta(days=1)).isocalendar()
    week_label = f"{iso_cal[0]}-W{iso_cal[1]:02d}"

    logger.info("=== Corma Feedback Pipeline (Daily) ===")
    logger.info(f"Date range: {from_date} to {to_date}")
    logger.info(f"Week label: {week_label}")
    logger.info(f"Dry run: {args.dry_run}")

    # Initialize clients
    leexi = LeexiClient(LEEXI_API_KEY_ID, LEEXI_API_KEY_SECRET)
    analyzer = TranscriptAnalyzer()
    notion = (
        NotionClient(NOTION_API_KEY, NOTION_DATABASE_ID, NOTION_WEEKLY_PARENT_PAGE_ID)
        if not args.dry_run
        else None
    )
    slack = (
        SlackClient(SLACK_WEBHOOK_URL, SLACK_WEEKLY_WEBHOOK_URL or None)
        if SLACK_WEBHOOK_URL and not args.dry_run
        else None
    )

    # Fetch call list
    logger.info("Fetching calls from Leexi...")
    raw_calls = leexi.get_calls(from_date, to_date)
    logger.info(f"Fetched {len(raw_calls)} calls")

    # Dedup check
    existing_titles: set[str] = set()
    if notion:
        existing_titles = notion.query_existing_entries(week_label)

    stats = {
        "total_calls": len(raw_calls),
        "calls_with_feedback_prompt": 0,
        "analyzed": 0,
        "skipped_no_prompt": 0,
        "skipped_empty": 0,
        "skipped_no_feedback": 0,
        "feedback_items_created": 0,
        "slack_notifications_sent": 0,
        "errors": 0,
    }
    all_results: list[tuple[CallMetadata, CallAnalysisResult]] = []

    for i, raw_call in enumerate(raw_calls):
        call_uuid = raw_call.get("uuid", "unknown")
        call_title = raw_call.get("title", "Untitled")
        logger.info(f"Processing call {i + 1}/{len(raw_calls)}: {call_uuid} ({call_title})")

        try:
            # Step 1: Fetch full call details (includes prompts)
            detail = leexi.get_call_details(call_uuid)

            # Step 2: Check for product feedback prompt
            prompt_output, has_prompt = leexi.extract_prompt_output(detail)

            if not has_prompt:
                logger.info("  Skipping: no product feedback prompt")
                stats["skipped_no_prompt"] += 1
                continue

            stats["calls_with_feedback_prompt"] += 1
            logger.info(f"  Found product feedback prompt ({len(prompt_output or '')} chars)")

            # Step 3: Build metadata with prompt output
            call = leexi.build_call_metadata(
                detail,
                feedback_prompt_output=prompt_output,
                has_feedback_prompt=True,
            )

            # Skip empty transcripts
            if not call.transcript_text or len(call.transcript_text.strip()) < 50:
                logger.info("  Skipping: empty/short transcript")
                stats["skipped_empty"] += 1
                continue

            # Step 4: Analyze with Claude (prompt output + transcript)
            result = analyzer.analyze_call(call)
            stats["analyzed"] += 1

            if result is None:
                logger.warning("  Analysis returned None")
                stats["errors"] += 1
                continue

            logger.info(
                f"  Call type: {result.call_type} | "
                f"Feedback items: {len(result.feedback_items)} | "
                f"MRR: {result.potential_mrr or 'N/A'}"
            )

            if args.dry_run:
                logger.info(f"  Summary: {result.call_summary}")
                for item in result.feedback_items:
                    logger.info(f"    [{item.priority}] [{item.category}] {item.title}")

            if not result.feedback_items:
                logger.info("  No product feedback extracted")
                stats["skipped_no_feedback"] += 1
                all_results.append((call, result))
                continue

            all_results.append((call, result))

            # Step 5: Create Notion entries
            if notion:
                for item in result.feedback_items:
                    if item.title in existing_titles:
                        logger.info(f"  Skipping duplicate: {item.title!r}")
                        continue
                    notion.create_feedback_entry(item, call, result, week_label)
                    existing_titles.add(item.title)
                    stats["feedback_items_created"] += 1

            # Step 6: Send Slack per-call notification
            if slack and result.feedback_items:
                # Use Leexi stage for call type, fall back to Claude's inference
                effective_call_type = call.leexi_call_type or result.call_type or "Other"
                # Use Leexi owner for Corma contact, fall back to Claude's participants
                effective_corma_contacts = (
                    [call.leexi_owner_name] if call.leexi_owner_name
                    else result.corma_participants
                )
                success = slack.send_call_feedback(
                    call_title=call.title or "Untitled",
                    call_date=(
                        call.performed_at[:10] if call.performed_at else "Unknown"
                    ),
                    call_type=effective_call_type,
                    customer_company=(
                        result.customer_company
                        or call.leexi_company_name
                        or call.leexi_contact_email
                        or "Unknown"
                    ),
                    corma_contacts=effective_corma_contacts,
                    feedback_items=[
                        {
                            "title": item.title,
                            "category": item.category,
                            "priority": item.priority,
                            "description": item.description,
                        }
                        for item in result.feedback_items
                    ],
                    potential_mrr=result.potential_mrr,
                    leexi_url=call.leexi_url,
                )
                if success:
                    stats["slack_notifications_sent"] += 1

        except Exception:
            logger.exception(f"Error processing call {call_uuid}")
            stats["errors"] += 1

    # Final stats
    logger.info("=== Pipeline Complete ===")
    logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
