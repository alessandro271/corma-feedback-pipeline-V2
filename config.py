import logging
import os

from dotenv import load_dotenv

load_dotenv(override=True)

# --- API Keys ---
LEEXI_API_KEY_ID = os.environ.get("LEEXI_API_KEY_ID", "")
LEEXI_API_KEY_SECRET = os.environ.get("LEEXI_API_KEY_SECRET", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NOTION_API_KEY = os.environ.get("NOTION_API_KEY", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
NOTION_WEEKLY_PARENT_PAGE_ID = os.environ.get("NOTION_WEEKLY_PARENT_PAGE_ID", "")

# --- Slack ---
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
SLACK_WEEKLY_WEBHOOK_URL = os.environ.get("SLACK_WEEKLY_WEBHOOK_URL", "")

# --- Leexi ---
LEEXI_BASE_URL = "https://public-api.leexi.ai/v1"
LEEXI_PAGE_SIZE = 100
LEEXI_RATE_LIMIT_PER_MIN = 50
FEEDBACK_PROMPT_TITLE = "product feedback prompt"

# --- Notion ---
NOTION_BASE_URL = "https://api.notion.com/v1"
NOTION_API_VERSION = "2022-06-28"
NOTION_MIN_REQUEST_INTERVAL = 0.34  # seconds, to stay under 3 req/sec

# --- Claude ---
CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
MAX_TRANSCRIPT_CHARS = 180_000

# --- Pipeline ---
DEFAULT_LOOKBACK_DAYS = 1  # Daily mode by default

# --- Corma Team ---
CORMA_EMAIL_DOMAIN = "corma.io"

CORMA_TEAM_NAMES: set[str] = {
    "heloise",
    "nikolai",
    "alessandro",
    "jean",
    "solal",
    "yann",
    "quentin",
    "louis",
    "samuel",
}

# --- Logging ---
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pipeline.log", encoding="utf-8"),
        ],
    )
    return logging.getLogger("corma-feedback")
