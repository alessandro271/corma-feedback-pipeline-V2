#!/usr/bin/env python3
"""Corma Customer Feedback Pipeline V2.

Fetches calls from Leexi, filters by the "product feedback prompt",
analyzes with Claude (enriched with Linear + Notion product context),
pushes to Notion, sends Slack notifications, and syncs to Linear.

Modes:
  - Default (daily): Process recent calls, create Notion entries, send Slack per-call,
                      sync to Linear, reply to Slack threads.
  - --weekly-digest: Query Notion for past 7 days and send a summary to Slack.
  - --update-schema: Add new properties (Status, MRR) to an existing Notion database.
  - --skip-linear:   Run daily pipeline but skip the Linear sync phase.
"""

import argparse
import json
import re
import sys
from datetime import date, timedelta

from config import (
    DEFAULT_LOOKBACK_DAYS,
    LEEXI_API_KEY_ID,
    LEEXI_API_KEY_SECRET,
    LINEAR_API_KEY,
    LINEAR_FEATURE_LABEL,
    LINEAR_INTEGRATION_LABEL,
    LINEAR_TEAM_ID,
    MAX_CONTEXT_CHARS,
    NOTION_API_KEY,
    NOTION_DATABASE_ID,
    NOTION_INTEGRATIONS_DB_ID,
    NOTION_WEEKLY_PARENT_PAGE_ID,
    SLACK_BOT_TOKEN,
    SLACK_CHANNEL_ID,
    SLACK_WEBHOOK_URL,
    SLACK_WEEKLY_WEBHOOK_URL,
    setup_logging,
)
from analyzer import TranscriptAnalyzer
from leexi_client import LeexiClient
from linear_client import LinearClient
from models import CallAnalysisResult, CallMetadata
from notion_client import NotionClient
from slack_client import SlackClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Corma Customer Feedback Pipeline V2"
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
        help="Analyze calls but do not write to Notion, Slack, or Linear.",
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
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        help="Skip the Linear sync phase (useful for debugging).",
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
        # Linear and Slack bot token are optional — warn if partially configured
        if LINEAR_API_KEY and not LINEAR_TEAM_ID:
            missing.append("LINEAR_TEAM_ID (required when LINEAR_API_KEY is set)")
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
        SlackClient(
            SLACK_WEBHOOK_URL,
            SLACK_WEEKLY_WEBHOOK_URL or None,
            SLACK_BOT_TOKEN or None,
            SLACK_CHANNEL_ID or None,
        )
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
    feedback_entries: list[dict] = []
    mrr_by_company: dict[str, str] = {}
    call_urls: set[str] = set()

    for page in pages:
        props = page.get("properties", {})

        title_prop = props.get("Feedback Title", {}).get("title", [])
        title = title_prop[0].get("plain_text", "") if title_prop else ""

        company_prop = props.get("Customer / Company", {}).get("rich_text", [])
        company = company_prop[0].get("plain_text", "") if company_prop else ""

        mrr_prop = props.get("MRR Potential", {}).get("rich_text", [])
        mrr = mrr_prop[0].get("plain_text", "") if mrr_prop else ""

        if title:
            feedback_entries.append({
                "title": title,
                "company": company or "Unknown",
                "mrr": mrr or "",
            })

        if mrr:
            company_key = company or "Unknown"
            if company_key not in mrr_by_company:
                mrr_by_company[company_key] = mrr

        url_prop = props.get("Source Call URL", {}).get("url")
        if url_prop:
            call_urls.add(url_prop)

    mrr_details = [{"company": c, "mrr": m} for c, m in mrr_by_company.items()]

    logger.info(
        f"Digest: {len(pages)} feedback items from {len(call_urls)} calls, "
        f"{len(mrr_details)} companies with MRR data"
    )

    if slack:
        slack.send_weekly_digest(
            week_label=week_label,
            total_calls_with_feedback=len(call_urls),
            total_feedback_items=len(pages),
            top_features=feedback_entries[:10],
            mrr_details=mrr_details,
            total_mrr=None,
        )
        logger.info("Weekly Slack digest sent")
    else:
        logger.info(
            f"DRY RUN: Would send digest with {len(pages)} items "
            f"from {len(call_urls)} calls"
        )


# ======================================================================
# Product Context Gathering
# ======================================================================


def gather_product_context(
    notion: NotionClient | None,
    linear: LinearClient | None,
    logger,
) -> str | None:
    """Build product context string from Linear + Notion integrations catalogue.

    Runs once at pipeline start. Returns formatted text for Claude's prompt.
    """
    sections: list[str] = []

    # 1. Fetch Linear issues/projects summary
    if linear:
        try:
            context = linear.build_context_summary()
            if context:
                sections.append(context)
                logger.info(f"Fetched Linear context ({len(context)} chars)")
        except Exception:
            logger.exception("Failed to fetch Linear context (non-fatal)")

    # 2. Fetch Notion integrations catalogue
    if notion and NOTION_INTEGRATIONS_DB_ID:
        try:
            integrations = notion.fetch_integrations_database(
                NOTION_INTEGRATIONS_DB_ID
            )
            if integrations:
                sections.append(
                    f"## Existing Integrations (from Notion catalogue)\n{integrations}"
                )
                logger.info(
                    f"Fetched integrations catalogue ({len(integrations)} chars)"
                )
        except Exception:
            logger.exception("Failed to fetch integrations catalogue (non-fatal)")

    if not sections:
        return None

    combined = "\n\n".join(sections)
    if len(combined) > MAX_CONTEXT_CHARS:
        combined = combined[:MAX_CONTEXT_CHARS] + "\n... (truncated)"

    return combined


# ======================================================================
# Linear Sync
# ======================================================================


def _parse_mrr(mrr_str: str | None) -> float | None:
    """Parse MRR string like '480 EUR' to float 480."""
    if not mrr_str:
        return None
    match = re.search(r"[\d,.]+", mrr_str)
    if match:
        try:
            return float(match.group().replace(",", ""))
        except ValueError:
            return None
    return None


def _parse_company_size(size_str: str | None) -> int | None:
    """Parse company size string like '150 users' to int 150."""
    if not size_str:
        return None
    match = re.search(r"\d+", size_str)
    if match:
        try:
            return int(match.group())
        except ValueError:
            return None
    return None


def _build_issue_description(item, call, result, company: str) -> str:
    """Build markdown description for a new Linear issue."""
    parts = [
        "## Customer Feedback",
        "",
        f"**Reported by:** {item.customer_name or company}",
        f"**Company:** {company}",
    ]
    if result.potential_mrr:
        parts.append(f"**MRR Potential:** {result.potential_mrr}")
    if result.company_size:
        parts.append(f"**Company Size:** {result.company_size}")
    if result.company_domain:
        parts.append(f"**Domain:** {result.company_domain}")
    parts.extend([
        f"**Call Type:** {result.call_type}",
        f"**Date:** {call.performed_at[:10] if call.performed_at else 'Unknown'}",
        "",
        "### Description",
        item.description,
        "",
        "### Verbatim Quote",
        f"> {item.verbatim_quote}",
    ])
    if call.leexi_url:
        parts.extend(["", "---", f"[View call in Leexi]({call.leexi_url})"])
    parts.extend(["", "*Auto-created by Corma Feedback Pipeline*"])
    return "\n".join(parts)


def sync_to_linear(
    linear: LinearClient,
    analyzer: TranscriptAnalyzer,
    notion: NotionClient | None,
    slack: SlackClient | None,
    all_results: list[tuple[CallMetadata, CallAnalysisResult]],
    logger,
) -> None:
    """Sync feedback items to Linear: create/update tickets and customer records."""
    logger.info("=== Linear Sync Phase ===")

    sync_stats = {
        "tickets_created": 0,
        "customer_needs_added": 0,
        "customers_created": 0,
        "customers_found": 0,
        "errors": 0,
    }

    # Pre-fetch label IDs
    integration_label_id = None
    feature_label_id = None
    try:
        integration_label_id = linear.get_or_create_label(LINEAR_INTEGRATION_LABEL)
        feature_label_id = linear.get_or_create_label(LINEAR_FEATURE_LABEL)
    except Exception:
        logger.exception("Failed to get/create labels")

    for call, result in all_results:
        if not result.feedback_items:
            continue

        company = (
            result.customer_company
            or call.leexi_company_name
            or "Unknown"
        )

        # --- Step A: Customer card (first, before any ticket work) ---
        customer_id = None
        customer_action = None
        if company != "Unknown":
            try:
                existing_customer = linear.find_customer(
                    domain=result.company_domain,
                    name=company,
                )
                if existing_customer:
                    customer_id = existing_customer["id"]
                    customer_action = "found"
                    # Update with latest data if available
                    linear.update_customer(
                        customer_id,
                        domain=result.company_domain,
                    )
                    sync_stats["customers_found"] += 1
                    logger.info(f"  Found existing customer: {company}")
                else:
                    # Create new customer card
                    new_customer = linear.create_customer(
                        name=company,
                        domain=result.company_domain,
                        revenue=_parse_mrr(result.potential_mrr),
                        size=_parse_company_size(result.company_size),
                    )
                    if new_customer:
                        customer_id = new_customer["id"]
                        customer_action = "created"
                        sync_stats["customers_created"] += 1
                        logger.info(f"  Created customer card: {company}")
            except Exception:
                logger.exception(f"  Failed to upsert customer: {company}")

        # --- Step B: Process each feedback item ---
        call_sync_results: list[dict] = []

        for item in result.feedback_items:
            try:
                sync_result = _sync_single_item(
                    linear, analyzer, item, call, result,
                    customer_id, integration_label_id, feature_label_id,
                    company, logger, sync_stats,
                )
                call_sync_results.append(sync_result)

                # Update Notion status
                if notion and item.notion_page_id and sync_result.get("action") != "error":
                    try:
                        notion.update_entry_status(
                            item.notion_page_id, "Added to Linear"
                        )
                    except Exception:
                        logger.warning(
                            f"  Failed to update Notion status for: {item.title}"
                        )
            except Exception:
                logger.exception(f"  Failed to sync item: {item.title}")
                sync_stats["errors"] += 1
                call_sync_results.append({
                    "feedback_title": item.title,
                    "action": "error",
                    "error_message": "unexpected error",
                })

        # --- Step C: Reply to Slack thread ---
        if slack and result.slack_thread_ts and call_sync_results:
            customer_summary = None
            if customer_action and company != "Unknown":
                mrr_part = f", {result.potential_mrr}" if result.potential_mrr else ""
                customer_summary = f"Customer: {company} ({customer_action}{mrr_part})"
            try:
                slack.send_linear_sync_reply(
                    thread_ts=result.slack_thread_ts,
                    sync_results=call_sync_results,
                    customer_summary=customer_summary,
                )
            except Exception:
                logger.warning("  Failed to send Slack thread reply")

    logger.info(f"Linear sync complete: {json.dumps(sync_stats)}")


def _sync_single_item(
    linear: LinearClient,
    analyzer: TranscriptAnalyzer,
    item,
    call: CallMetadata,
    result: CallAnalysisResult,
    customer_id: str | None,
    integration_label_id: str | None,
    feature_label_id: str | None,
    company: str,
    logger,
    sync_stats: dict,
) -> dict:
    """Sync a single feedback item to Linear. Returns a result dict."""

    # Search for matching existing tickets
    candidates = linear.find_matching_issue(item.title, item.category)

    matched_issue = None
    if candidates:
        # Claude-assisted matching if multiple candidates
        matched_issue = analyzer.pick_best_match(
            item.title, item.description, candidates
        )

    if matched_issue:
        # --- Existing ticket found: add customer need (not a comment) ---
        if customer_id:
            linear.create_customer_need(
                issue_id=matched_issue["id"],
                customer_id=customer_id,
                body=f"{item.title}: {item.description}",
            )
        logger.info(
            f"  Added customer need to {matched_issue['identifier']}: {item.title}"
        )
        sync_stats["customer_needs_added"] += 1
        return {
            "feedback_title": item.title,
            "action": "customer_need_added",
            "issue_identifier": matched_issue.get("identifier"),
            "issue_url": matched_issue.get("url"),
        }
    else:
        # --- No match: create new ticket in Backlog with appropriate label ---
        label_ids = []
        project_id = None

        if item.category == "Integration Request" and integration_label_id:
            label_ids.append(integration_label_id)
        else:
            # Feature Request or other categories get the Feature label
            if feature_label_id:
                label_ids.append(feature_label_id)

            # Try to find a matching project for feature requests
            try:
                project_candidates = linear.find_matching_project(item.title)
                if project_candidates:
                    matched_project = analyzer.pick_best_match(
                        item.title, item.description, project_candidates
                    )
                    if matched_project:
                        project_id = matched_project.get("id")
                        logger.info(
                            f"  Linking to project: {matched_project.get('name')}"
                        )
            except Exception:
                logger.warning("  Failed to search for matching project")

        priority_map = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
        priority = priority_map.get(item.priority, 3)

        description = _build_issue_description(item, call, result, company)

        created = linear.create_issue(
            title=item.title,
            description=description,
            label_ids=label_ids,
            project_id=project_id,
            priority=priority,
        )

        if created:
            # Attach customer need to the new issue
            if customer_id:
                linear.create_customer_need(
                    issue_id=created["id"],
                    customer_id=customer_id,
                    body=f"{item.title}: {item.description}",
                )
            logger.info(
                f"  Created Linear issue {created['identifier']}: {item.title}"
            )
            sync_stats["tickets_created"] += 1
            return {
                "feedback_title": item.title,
                "action": "created",
                "issue_identifier": created.get("identifier"),
                "issue_url": created.get("url"),
            }
        else:
            sync_stats["errors"] += 1
            return {
                "feedback_title": item.title,
                "action": "error",
                "error_message": "failed to create issue",
            }


# ======================================================================
# Daily Pipeline
# ======================================================================


def run_daily_pipeline(args, logger) -> None:
    """Main daily pipeline: fetch calls, analyze, store, notify, sync to Linear."""
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

    logger.info("=== Corma Feedback Pipeline V2 (Daily) ===")
    logger.info(f"Date range: {from_date} to {to_date}")
    logger.info(f"Week label: {week_label}")
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Skip Linear: {args.skip_linear}")

    # Initialize clients
    # Note: Notion and Linear are always initialised (when configured) because
    # they're needed for read-only context gathering even in dry-run mode.
    # The dry-run flag controls whether we *write* to Notion/Slack/Linear.
    leexi = LeexiClient(LEEXI_API_KEY_ID, LEEXI_API_KEY_SECRET)
    analyzer = TranscriptAnalyzer()
    notion = (
        NotionClient(NOTION_API_KEY, NOTION_DATABASE_ID, NOTION_WEEKLY_PARENT_PAGE_ID)
        if NOTION_API_KEY and NOTION_DATABASE_ID
        else None
    )
    slack = (
        SlackClient(
            SLACK_WEBHOOK_URL,
            SLACK_WEEKLY_WEBHOOK_URL or None,
            SLACK_BOT_TOKEN or None,
            SLACK_CHANNEL_ID or None,
        )
        if SLACK_WEBHOOK_URL and not args.dry_run
        else None
    )
    linear = None
    if LINEAR_API_KEY and LINEAR_TEAM_ID:
        linear = LinearClient(LINEAR_API_KEY, LINEAR_TEAM_ID)
        logger.info("Linear integration enabled")

    # --- Phase 1: Gather product context (once, before processing calls) ---
    product_context = gather_product_context(notion, linear, logger)
    if product_context:
        logger.info(f"Product context loaded ({len(product_context)} chars)")
    else:
        logger.info("No product context available")

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

            # Step 4: Analyze with Claude (with product context)
            result = analyzer.analyze_call(call, product_context=product_context)
            stats["analyzed"] += 1

            if result is None:
                logger.warning("  Analysis returned None")
                stats["errors"] += 1
                continue

            logger.info(
                f"  Call type: {result.call_type} | "
                f"Feedback items: {len(result.feedback_items)} | "
                f"MRR: {result.potential_mrr or 'N/A'} | "
                f"Size: {result.company_size or 'N/A'} | "
                f"Domain: {result.company_domain or 'N/A'}"
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

            # Step 5: Create Notion entries (track page IDs for Linear sync)
            if notion and not args.dry_run:
                for item in result.feedback_items:
                    if item.title in existing_titles:
                        logger.info(f"  Skipping duplicate: {item.title!r}")
                        continue
                    page_id = notion.create_feedback_entry(item, call, result, week_label)
                    item.notion_page_id = page_id
                    existing_titles.add(item.title)
                    stats["feedback_items_created"] += 1

            # Step 6: Send Slack per-call notification (capture ts for threading)
            if slack and result.feedback_items:
                effective_call_type = call.leexi_call_type or result.call_type or "Other"
                effective_corma_contacts = (
                    [call.leexi_owner_name] if call.leexi_owner_name
                    else result.corma_participants
                )
                slack_ts = slack.send_call_feedback(
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
                    company_size=result.company_size,
                    company_domain=result.company_domain,
                )
                if slack_ts:
                    result.slack_thread_ts = slack_ts
                stats["slack_notifications_sent"] += 1

        except Exception:
            logger.exception(f"Error processing call {call_uuid}")
            stats["errors"] += 1

    # --- Phase 2: Linear sync (after all calls are processed) ---
    if linear and not args.dry_run and not args.skip_linear:
        try:
            sync_to_linear(linear, analyzer, notion, slack, all_results, logger)
        except Exception:
            logger.exception("Linear sync phase failed")

    # Final stats
    logger.info("=== Pipeline Complete ===")
    logger.info(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
