#!/usr/bin/env python3
"""One-time setup script to create the Notion database and weekly summary parent page.

Prerequisites:
  1. Create a Notion internal integration at https://www.notion.so/my-integrations
     - Give it a name like "Corma Feedback Pipeline"
     - Select your workspace
     - Copy the "Internal Integration Secret"

  2. In Notion, create (or pick) a page where the database and weekly summaries will live.
     - Open that page, click "..." menu → "Connections" → add your integration.
     - Copy the page ID from the URL:
       https://www.notion.so/yourworkspace/PAGE_TITLE-<page_id>
       The page_id is the 32-char hex string at the end (add dashes if needed).

  3. Set environment variables:
       NOTION_API_KEY=<your integration secret>
       NOTION_PARENT_PAGE_ID=<the page id from step 2>

  4. Run: python setup_notion.py

  The script will print the DATABASE_ID and WEEKLY_PARENT_PAGE_ID to put in your .env.
"""

import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "")
NOTION_PARENT_PAGE_ID = os.environ.get("NOTION_PARENT_PAGE_ID", "")
NOTION_API_VERSION = "2022-06-28"
BASE_URL = "https://api.notion.com/v1"


def notion_request(method: str, endpoint: str, json_body: dict | None = None) -> dict:
    headers = {
        "Authorization": f"Bearer {NOTION_API_KEY}",
        "Notion-Version": NOTION_API_VERSION,
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    resp = requests.request(method, url, headers=headers, json=json_body, timeout=30)
    if not resp.ok:
        print(f"Error {resp.status_code}: {resp.text}")
        sys.exit(1)
    return resp.json()


def create_feedback_database(parent_page_id: str) -> str:
    """Create the Product Feedback database and return its ID."""
    body = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": "Product Feedback"}}],
        "is_inline": False,
        "properties": {
            "Feedback Title": {"title": {}},
            "Category": {
                "select": {
                    "options": [
                        {"name": "Feature Request", "color": "blue"},
                        {"name": "Bug Report", "color": "red"},
                        {"name": "UX Issue", "color": "orange"},
                        {"name": "Integration Request", "color": "purple"},
                        {"name": "Performance Issue", "color": "yellow"},
                        {"name": "Pricing Feedback", "color": "green"},
                        {"name": "Other", "color": "gray"},
                    ]
                }
            },
            "Description": {"rich_text": {}},
            "Verbatim Quote": {"rich_text": {}},
            "Customer / Company": {"rich_text": {}},
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
            "Date": {"date": {}},
            "Corma Contact": {"rich_text": {}},
            "Priority": {
                "select": {
                    "options": [
                        {"name": "Critical", "color": "red"},
                        {"name": "High", "color": "orange"},
                        {"name": "Medium", "color": "yellow"},
                        {"name": "Low", "color": "gray"},
                    ]
                }
            },
            "Sentiment": {
                "select": {
                    "options": [
                        {"name": "Positive", "color": "green"},
                        {"name": "Neutral", "color": "gray"},
                        {"name": "Negative", "color": "orange"},
                        {"name": "Frustrated", "color": "red"},
                    ]
                }
            },
            "Source Call URL": {"url": {}},
            "Week": {"rich_text": {}},
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
        },
    }
    data = notion_request("POST", "databases", json_body=body)
    return data["id"]


def create_weekly_summaries_page(parent_page_id: str) -> str:
    """Create a parent page for weekly summaries and return its ID."""
    body = {
        "parent": {"page_id": parent_page_id},
        "properties": {
            "title": [
                {
                    "type": "text",
                    "text": {"content": "Weekly Feedback Summaries"},
                }
            ]
        },
        "children": [
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [
                        {
                            "type": "text",
                            "text": {
                                "content": "Weekly summary pages will be created "
                                "automatically under this page by the feedback pipeline."
                            },
                        }
                    ]
                },
            }
        ],
    }
    data = notion_request("POST", "pages", json_body=body)
    return data["id"]


def main() -> None:
    if not NOTION_API_KEY:
        print("Error: NOTION_API_KEY environment variable is not set.")
        print("Set it to your Notion internal integration secret.")
        sys.exit(1)

    if not NOTION_PARENT_PAGE_ID:
        print("Error: NOTION_PARENT_PAGE_ID environment variable is not set.")
        print("Set it to the Notion page ID where you want the database created.")
        print(
            "You can find the page ID in the URL: "
            "https://www.notion.so/workspace/Page-Title-<page_id>"
        )
        sys.exit(1)

    print(f"Using parent page: {NOTION_PARENT_PAGE_ID}")
    print()

    print("Creating Product Feedback database...")
    db_id = create_feedback_database(NOTION_PARENT_PAGE_ID)
    print(f"  Database created: {db_id}")
    print()

    print("Creating Weekly Summaries page...")
    weekly_id = create_weekly_summaries_page(NOTION_PARENT_PAGE_ID)
    print(f"  Weekly summaries page created: {weekly_id}")
    print()

    print("=" * 60)
    print("Setup complete! Add these to your .env file:")
    print()
    print(f"NOTION_DATABASE_ID={db_id}")
    print(f"NOTION_WEEKLY_PARENT_PAGE_ID={weekly_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
