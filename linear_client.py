"""Linear GraphQL API client for the Corma Feedback Pipeline.

Provides context gathering (issues, projects), issue search & CRUD,
and customer management with smart deduplication.
"""

import logging
import re
import unicodedata
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("corma-feedback.linear")

GRAPHQL_ENDPOINT = "https://api.linear.app/graphql"

# ---------------------------------------------------------------------------
# GraphQL queries & mutations
# ---------------------------------------------------------------------------

QUERY_ISSUES_SUMMARY = """
query IssuesSummary($after: String) {
    issues(
        first: 250,
        after: $after,
        filter: { state: { type: { nin: ["canceled"] } } }
    ) {
        pageInfo { hasNextPage endCursor }
        nodes {
            id identifier title
            state { name type }
            project { id name }
            labels { nodes { name } }
        }
    }
}
"""

QUERY_PROJECTS = """
query Projects {
    projects(first: 100) {
        nodes { id name state description }
    }
}
"""

QUERY_SEARCH_ISSUES = """
query SearchIssues($term: String!, $first: Int!) {
    searchIssues(first: $first, term: $term) {
        nodes {
            id identifier title url
            state { name type }
            project { id name }
            labels { nodes { name } }
        }
    }
}
"""

QUERY_FILTER_ISSUES = """
query FilterIssues($titleFilter: String!, $first: Int!) {
    issues(
        first: $first,
        filter: { title: { containsIgnoreCase: $titleFilter } }
    ) {
        nodes {
            id identifier title url
            state { name type }
            project { id name }
            labels { nodes { name } }
        }
    }
}
"""

QUERY_TEAM_STATES = """
query TeamStates($teamId: String!) {
    team(id: $teamId) {
        states {
            nodes { id name type }
        }
    }
}
"""

QUERY_LABELS = """
query Labels($teamId: String!) {
    team(id: $teamId) {
        labels {
            nodes { id name }
        }
    }
}
"""

MUTATION_CREATE_ISSUE = """
mutation IssueCreate($input: IssueCreateInput!) {
    issueCreate(input: $input) {
        success
        issue { id identifier title url }
    }
}
"""

MUTATION_CREATE_LABEL = """
mutation LabelCreate($input: IssueLabelCreateInput!) {
    issueLabelCreate(input: $input) {
        success
        issueLabel { id name }
    }
}
"""

MUTATION_CREATE_COMMENT = """
mutation CommentCreate($input: CommentCreateInput!) {
    commentCreate(input: $input) {
        success
        comment { id }
    }
}
"""

# Customer queries & mutations
QUERY_CUSTOMERS = """
query Customers($after: String) {
    customers(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes { id name domains externalIds }
    }
}
"""

MUTATION_CREATE_CUSTOMER = """
mutation CustomerCreate($input: CustomerCreateInput!) {
    customerCreate(input: $input) {
        success
        customer { id name domains }
    }
}
"""

MUTATION_UPDATE_CUSTOMER = """
mutation CustomerUpdate($id: String!, $input: CustomerUpdateInput!) {
    customerUpdate(id: $id, input: $input) {
        success
        customer { id name }
    }
}
"""

QUERY_CUSTOMER_STATUSES = """
query CustomerStatuses {
    customerStatuses {
        nodes { id name }
    }
}
"""

MUTATION_CREATE_CUSTOMER_NEED = """
mutation CustomerNeedCreate($input: CustomerNeedCreateInput!) {
    customerNeedCreate(input: $input) {
        success
    }
}
"""

# Legal suffixes to strip for fuzzy company name matching
_LEGAL_SUFFIXES = re.compile(
    r"\b(sas|sarl|sa|gmbh|ag|inc|incorporated|llc|ltd|limited|"
    r"corp|corporation|plc|bv|nv|pty|co|company)\b",
    re.IGNORECASE,
)

# Common prefixes to strip from feedback titles for search
_TITLE_PREFIXES = [
    "integration with ", "integration for ", "integrate with ",
    "support for ", "add ", "implement ", "create ", "build ",
    "intégration avec ", "intégration de ",
]
_TITLE_SUFFIXES = [
    " integration", " support", " feature", " request",
]


def _is_retryable(exc: BaseException) -> bool:
    """Check if the exception warrants a retry (rate limit or server error)."""
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else 0
        return status == 429 or status >= 500
    if isinstance(exc, (requests.ConnectionError, requests.Timeout)):
        return True
    return False


def _normalize_company_name(name: str) -> str:
    """Normalize a company name for fuzzy matching.

    Strips legal suffixes, accents, whitespace/hyphens, and lowercases.
    """
    # Decompose unicode and strip accent marks
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Lowercase
    normalized = ascii_name.lower().strip()
    # Remove hyphens, dots, colons, and other punctuation
    normalized = re.sub(r"[\-\.\:\;\,\(\)\[\]\"\']+", "", normalized)
    # Strip legal suffixes (after punctuation removal to avoid false matches)
    normalized = _LEGAL_SUFFIXES.sub("", normalized)
    # Collapse whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_search_terms(feedback_title: str) -> str:
    """Extract distinctive keywords from a feedback title for Linear search."""
    title = feedback_title.strip().lower()
    for prefix in _TITLE_PREFIXES:
        if title.startswith(prefix):
            title = title[len(prefix):]
            break
    for suffix in _TITLE_SUFFIXES:
        if title.endswith(suffix):
            title = title[: -len(suffix)]
            break
    return title.strip()


def _filter_by_title_relevance(
    candidates: list[dict], search_terms: str
) -> list[dict]:
    """Keep only candidates whose title contains a significant search term.

    Linear's searchIssues uses semantic matching that often returns unrelated
    results (e.g. "RAMP" → "Akuiteo Integration"). This filter ensures at
    least one keyword from the search terms appears in the candidate title.
    """
    if not search_terms:
        return candidates
    terms = [t for t in search_terms.lower().split() if len(t) >= 3]
    if not terms:
        return candidates
    return [
        c for c in candidates
        if any(term in c.get("title", "").lower() for term in terms)
    ]


def _get_logo_url(domain: str | None) -> str | None:
    """Build a logo URL from the company domain.

    Note: Linear's logoUrl field only accepts https://public.linear.app URLs
    (uploaded via Linear's asset system). External URLs are rejected.
    Returns None — logo must be set manually or via Linear's upload API.
    """
    # Linear rejects external URLs for logoUrl (requires public.linear.app host).
    # Clearbit/favicon URLs cannot be used directly.
    return None


class LinearClient:
    """Client for Linear's GraphQL API."""

    def __init__(self, api_key: str, team_id: str) -> None:
        self.team_id = team_id
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": api_key,
            "Content-Type": "application/json",
        })
        # Caches (populated lazily)
        self._backlog_state_id: str | None = None
        self._label_cache: dict[str, str] = {}  # name -> id
        self._all_customers: list[dict] | None = None
        self._customer_status_cache: dict[str, str] = {}  # name -> id

    # ------------------------------------------------------------------
    # Core GraphQL helper
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=1, max=30),
        reraise=True,
    )
    def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict:
        """Execute a GraphQL query/mutation with retry logic."""
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        resp = self.session.post(GRAPHQL_ENDPOINT, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            error_msg = "; ".join(e.get("message", "") for e in data["errors"])
            logger.error(f"Linear GraphQL error: {error_msg}")
            raise RuntimeError(f"Linear GraphQL error: {error_msg}")
        return data.get("data", {})

    # ------------------------------------------------------------------
    # Phase 1: Context gathering
    # ------------------------------------------------------------------

    def fetch_issues_summary(self) -> list[dict]:
        """Fetch all non-cancelled issues with title, state, project, labels."""
        all_issues: list[dict] = []
        cursor: str | None = None
        while True:
            variables: dict[str, Any] = {}
            if cursor:
                variables["after"] = cursor
            data = self._graphql(QUERY_ISSUES_SUMMARY, variables)
            issues_data = data.get("issues", {})
            nodes = issues_data.get("nodes", [])
            all_issues.extend(nodes)
            page_info = issues_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
        logger.info(f"Fetched {len(all_issues)} issues from Linear")
        return all_issues

    def fetch_projects(self) -> list[dict]:
        """Fetch all projects with name, state, description."""
        data = self._graphql(QUERY_PROJECTS)
        projects = data.get("projects", {}).get("nodes", [])
        logger.info(f"Fetched {len(projects)} projects from Linear")
        return projects

    def build_context_summary(self) -> str:
        """Build a concise text summary of existing features/projects for Claude."""
        issues = self.fetch_issues_summary()
        projects = self.fetch_projects()

        sections: list[str] = []

        # Active projects
        active_projects = [
            p for p in projects if p.get("state") in ("planned", "started")
        ]
        if active_projects:
            lines = [f"- {p['name']} ({p['state']})" for p in active_projects]
            sections.append(
                "### Active Projects:\n" + "\n".join(lines)
            )

        # Split issues into integration vs feature tickets
        integration_issues: list[str] = []
        feature_issues: list[str] = []

        for issue in issues:
            state = issue.get("state", {})
            state_name = state.get("name", "Unknown")
            state_type = state.get("type", "")
            title = issue.get("title", "")
            labels = [l["name"] for l in issue.get("labels", {}).get("nodes", [])]

            # Skip completed items older than what we need for context
            line = f"- [{state_name}] {title}"

            if any("integration" in l.lower() for l in labels):
                integration_issues.append(line)
            elif state_type not in ("completed",):
                # Include active/backlog feature tickets
                feature_issues.append(line)

        if integration_issues:
            sections.append(
                "### Integration Tickets:\n"
                + "\n".join(integration_issues[:100])  # Cap to keep context manageable
            )

        if feature_issues:
            sections.append(
                "### Feature Tickets (active/backlog):\n"
                + "\n".join(feature_issues[:200])  # Cap
            )

        if not sections:
            return ""

        return "## Linear Development Tickets\n\n" + "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Phase 3: Issue search & matching
    # ------------------------------------------------------------------

    def search_issues(self, term: str, limit: int = 10) -> list[dict]:
        """Full-text search using Linear's searchIssues."""
        data = self._graphql(
            QUERY_SEARCH_ISSUES, {"term": term, "first": limit}
        )
        return data.get("searchIssues", {}).get("nodes", [])

    def find_issues_by_title(self, title_filter: str, limit: int = 10) -> list[dict]:
        """Filter issues by title substring (case-insensitive)."""
        data = self._graphql(
            QUERY_FILTER_ISSUES, {"titleFilter": title_filter, "first": limit}
        )
        return data.get("issues", {}).get("nodes", [])

    def find_matching_issue(
        self, feedback_title: str, category: str
    ) -> list[dict]:
        """Find Linear issues matching the feedback title.

        Returns a list of candidates (may be empty, one, or multiple).
        The caller should use Claude-assisted matching when len > 1.
        Results are filtered by title relevance to avoid false matches
        from Linear's semantic search.
        """
        key_terms = _extract_search_terms(feedback_title)

        # Tier 1: Full-text search with complete title
        results = self.search_issues(feedback_title, limit=5)
        results = _filter_by_title_relevance(results, key_terms)
        if results:
            return results

        # Tier 2: Extract key terms and search again
        if key_terms and key_terms != feedback_title.lower():
            results = self.search_issues(key_terms, limit=5)
            results = _filter_by_title_relevance(results, key_terms)
            if results:
                return results

        # Tier 3: containsIgnoreCase filter on key terms
        if key_terms:
            results = self.find_issues_by_title(key_terms, limit=5)
            if results:
                return results

        return []

    def find_matching_project(self, feedback_title: str) -> list[dict]:
        """Find projects whose name relates to the feedback title.

        Returns candidates for Claude-assisted matching.
        """
        projects = self.fetch_projects()
        active = [p for p in projects if p.get("state") in ("planned", "started")]
        if not active:
            return []

        key_terms = _extract_search_terms(feedback_title).lower()
        if not key_terms:
            return active  # Return all for Claude to decide

        # Simple keyword overlap filter
        candidates = []
        terms_words = set(key_terms.split())
        for project in active:
            project_words = set(project.get("name", "").lower().split())
            if terms_words & project_words:
                candidates.append(project)

        # If no keyword overlap, return all active projects for Claude to assess
        return candidates if candidates else active

    # ------------------------------------------------------------------
    # Phase 3: Issue CRUD
    # ------------------------------------------------------------------

    def get_backlog_state_id(self) -> str | None:
        """Get the 'Backlog' workflow state ID for the team (cached)."""
        if self._backlog_state_id:
            return self._backlog_state_id
        data = self._graphql(QUERY_TEAM_STATES, {"teamId": self.team_id})
        states = data.get("team", {}).get("states", {}).get("nodes", [])
        for state in states:
            if state.get("type") == "backlog":
                self._backlog_state_id = state["id"]
                return self._backlog_state_id
        # Fallback: look for a state named "Backlog"
        for state in states:
            if state.get("name", "").lower() == "backlog":
                self._backlog_state_id = state["id"]
                return self._backlog_state_id
        logger.warning("No backlog state found for team")
        return None

    def get_or_create_label(self, name: str, color: str = "#6B7280") -> str | None:
        """Get an existing label by name or create it. Returns label ID."""
        # Check cache first
        if name in self._label_cache:
            return self._label_cache[name]

        # Query team labels
        data = self._graphql(QUERY_LABELS, {"teamId": self.team_id})
        labels = data.get("team", {}).get("labels", {}).get("nodes", [])
        for label in labels:
            self._label_cache[label["name"]] = label["id"]
            if label["name"].lower() == name.lower():
                return label["id"]

        # Create new label
        result = self._graphql(MUTATION_CREATE_LABEL, {
            "input": {"name": name, "color": color, "teamId": self.team_id}
        })
        created = result.get("issueLabelCreate", {})
        if created.get("success"):
            label_id = created["issueLabel"]["id"]
            self._label_cache[name] = label_id
            logger.info(f"Created Linear label: {name}")
            return label_id

        logger.error(f"Failed to create label: {name}")
        return None

    def create_issue(
        self,
        title: str,
        description: str,
        label_ids: list[str] | None = None,
        project_id: str | None = None,
        priority: int = 0,
    ) -> dict | None:
        """Create a new Linear issue in Backlog state."""
        input_data: dict[str, Any] = {
            "title": title,
            "description": description,
            "teamId": self.team_id,
            "priority": priority,
        }
        # Set to Backlog state
        backlog_id = self.get_backlog_state_id()
        if backlog_id:
            input_data["stateId"] = backlog_id
        if label_ids:
            input_data["labelIds"] = label_ids
        if project_id:
            input_data["projectId"] = project_id

        result = self._graphql(MUTATION_CREATE_ISSUE, {"input": input_data})
        issue_data = result.get("issueCreate", {})
        if issue_data.get("success"):
            issue = issue_data["issue"]
            logger.info(f"Created Linear issue: {issue['identifier']} — {title}")
            return issue

        logger.error(f"Failed to create issue: {title}")
        return None

    def add_comment(self, issue_id: str, body: str) -> bool:
        """Add a comment to an existing issue."""
        result = self._graphql(
            MUTATION_CREATE_COMMENT,
            {"input": {"issueId": issue_id, "body": body}},
        )
        return result.get("commentCreate", {}).get("success", False)

    # ------------------------------------------------------------------
    # Phase 3: Customer management
    # ------------------------------------------------------------------

    def _fetch_all_customers(self) -> list[dict]:
        """Fetch all customers (cached after first call)."""
        if self._all_customers is not None:
            return self._all_customers

        all_customers: list[dict] = []
        cursor: str | None = None
        while True:
            variables: dict[str, Any] = {}
            if cursor:
                variables["after"] = cursor
            data = self._graphql(QUERY_CUSTOMERS, variables)
            customers_data = data.get("customers", {})
            nodes = customers_data.get("nodes", [])
            all_customers.extend(nodes)
            page_info = customers_data.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")

        self._all_customers = all_customers
        logger.info(f"Fetched {len(all_customers)} customers from Linear")
        return all_customers

    def find_customer(
        self, domain: str | None = None, name: str | None = None
    ) -> dict | None:
        """Find an existing customer using multi-strategy deduplication.

        Tries: 1) domain match, 2) exact name, 3) fuzzy name.
        """
        customers = self._fetch_all_customers()
        if not customers:
            return None

        # Strategy 1: Domain match (most reliable)
        if domain:
            domain_lower = domain.lower().strip()
            for cust in customers:
                cust_domains = cust.get("domains", []) or []
                if any(d.lower().strip() == domain_lower for d in cust_domains):
                    logger.info(
                        f"Customer matched by domain '{domain}': {cust['name']}"
                    )
                    return cust

        # Strategy 2: Exact name match (case-insensitive)
        if name:
            name_lower = name.lower().strip()
            for cust in customers:
                if cust.get("name", "").lower().strip() == name_lower:
                    logger.info(
                        f"Customer matched by exact name: {cust['name']}"
                    )
                    return cust

        # Strategy 3: Fuzzy name match (exact after normalization)
        if name:
            normalized_input = _normalize_company_name(name)
            if normalized_input:
                for cust in customers:
                    normalized_existing = _normalize_company_name(
                        cust.get("name", "")
                    )
                    if normalized_existing and normalized_input == normalized_existing:
                        logger.info(
                            f"Customer matched by fuzzy name: "
                            f"'{name}' ≈ '{cust['name']}'"
                        )
                        return cust

        # Strategy 4: Containment match — handles "Lucca - Nouvel élément : Deal" matching "Lucca"
        if name:
            normalized_input = _normalize_company_name(name)
            if normalized_input and len(normalized_input) >= 3:
                for cust in customers:
                    normalized_existing = _normalize_company_name(
                        cust.get("name", "")
                    )
                    if not normalized_existing or len(normalized_existing) < 3:
                        continue
                    # Check if one fully contains the other (as a word boundary)
                    if (
                        normalized_existing in normalized_input.split()
                        or normalized_input in normalized_existing.split()
                        or normalized_input.startswith(normalized_existing + " ")
                        or normalized_existing.startswith(normalized_input + " ")
                    ):
                        logger.info(
                            f"Customer matched by containment: "
                            f"'{name}' ⊇ '{cust['name']}'"
                        )
                        return cust

        return None

    def get_customer_status_id(self, status_name: str) -> str | None:
        """Resolve a customer status name (e.g. 'Active', 'Prospect') to its ID."""
        if not self._customer_status_cache:
            try:
                result = self._graphql(QUERY_CUSTOMER_STATUSES, {})
                for status in result.get("customerStatuses", {}).get("nodes", []):
                    self._customer_status_cache[status["name"].lower()] = status["id"]
            except Exception:
                logger.warning("Failed to fetch customer statuses from Linear")
                return None
        return self._customer_status_cache.get(status_name.lower())

    def create_customer(
        self,
        name: str,
        domain: str | None = None,
        revenue: float | None = None,
        size: int | None = None,
        status: str | None = None,
    ) -> dict | None:
        """Create a new customer card in Linear."""
        input_data: dict[str, Any] = {"name": name}
        if domain:
            input_data["domains"] = [domain]
            logo_url = _get_logo_url(domain)
            if logo_url:
                input_data["logoUrl"] = logo_url
        if revenue is not None:
            input_data["revenue"] = int(revenue)
        if size is not None:
            input_data["size"] = int(size)
        if status:
            status_id = self.get_customer_status_id(status)
            if status_id:
                input_data["statusId"] = status_id

        result = self._graphql(MUTATION_CREATE_CUSTOMER, {"input": input_data})
        customer_data = result.get("customerCreate", {})
        if customer_data.get("success"):
            customer = customer_data["customer"]
            logger.info(f"Created Linear customer: {customer['name']}")
            # Invalidate cache
            self._all_customers = None
            return customer

        logger.error(f"Failed to create customer: {name}")
        return None

    def update_customer(
        self,
        customer_id: str,
        domain: str | None = None,
        revenue: float | None = None,
        size: int | None = None,
    ) -> bool:
        """Update an existing customer with new data."""
        input_data: dict[str, Any] = {}
        if domain:
            input_data["domains"] = [domain]
            logo_url = _get_logo_url(domain)
            if logo_url:
                input_data["logoUrl"] = logo_url
        if revenue is not None:
            input_data["revenue"] = int(revenue)
        if size is not None:
            input_data["size"] = int(size)

        if not input_data:
            return True  # Nothing to update

        result = self._graphql(
            MUTATION_UPDATE_CUSTOMER,
            {"id": customer_id, "input": input_data},
        )
        success = result.get("customerUpdate", {}).get("success", False)
        if success:
            self._all_customers = None  # Invalidate cache
        return success

    def create_customer_need(
        self, issue_id: str, customer_id: str, body: str
    ) -> bool:
        """Attach a customer request/need to an issue."""
        result = self._graphql(
            MUTATION_CREATE_CUSTOMER_NEED,
            {"input": {"issueId": issue_id, "customerId": customer_id, "body": body}},
        )
        return result.get("customerNeedCreate", {}).get("success", False)
