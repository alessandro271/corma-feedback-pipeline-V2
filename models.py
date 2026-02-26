from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Speaker(BaseModel):
    name: str
    email: Optional[str] = None
    is_corma_team: bool = False
    speaker_index: int = 0


class CallMetadata(BaseModel):
    uuid: str
    title: Optional[str] = None
    duration: Optional[float] = None
    performed_at: Optional[str] = None
    direction: Optional[str] = None
    speakers: list[Speaker] = Field(default_factory=list)
    customer_emails: list[str] = Field(default_factory=list)
    transcript_text: Optional[str] = None
    leexi_url: Optional[str] = None
    feedback_prompt_output: Optional[str] = Field(
        default=None,
        description="Structured output from the 'product feedback prompt' in Leexi",
    )
    has_feedback_prompt: bool = Field(
        default=False,
        description="Whether this call had the product feedback prompt executed",
    )
    leexi_company_name: Optional[str] = Field(
        default=None,
        description="Company name extracted from Leexi deal/contact info",
    )
    leexi_contact_email: Optional[str] = Field(
        default=None,
        description="Primary external contact email from Leexi",
    )
    leexi_call_type: Optional[str] = Field(
        default=None,
        description="Call type derived from Leexi deal stage (Demo, Customer Success, Discovery, Support)",
    )
    leexi_owner_name: Optional[str] = Field(
        default=None,
        description="Corma team member who owns this call in Leexi",
    )


class FeedbackItem(BaseModel):
    title: str = Field(description="Short summary of the feedback (max 80 chars)")
    category: str = Field(
        description="One of: Feature Request, Bug Report, UX Issue, "
        "Integration Request, Performance Issue, Pricing Feedback, Other"
    )
    description: str = Field(description="1-3 sentence explanation of the feedback")
    verbatim_quote: str = Field(
        description="Exact quote from the customer in the transcript"
    )
    sentiment: str = Field(description="One of: Positive, Neutral, Negative, Frustrated")
    priority: str = Field(description="One of: Critical, High, Medium, Low")
    customer_company: Optional[str] = Field(
        default=None, description="Inferred company name if mentioned"
    )
    customer_name: Optional[str] = Field(
        default=None, description="Name of the customer/prospect speaker"
    )


class CallAnalysisResult(BaseModel):
    call_type: str = Field(
        description="One of: Demo, Customer Success, Discovery, Support, Other"
    )
    is_external_call: bool = Field(
        description="True if at least one non-Corma participant is present"
    )
    corma_participants: list[str] = Field(
        default_factory=list, description="Names of Corma team members on the call"
    )
    customer_participants: list[str] = Field(
        default_factory=list, description="Names of external participants"
    )
    customer_company: Optional[str] = Field(
        default=None, description="Inferred company name if mentioned"
    )
    feedback_items: list[FeedbackItem] = Field(
        default_factory=list, description="Extracted product feedback items"
    )
    call_summary: str = Field(
        description="2-3 sentence summary of the call"
    )
    potential_mrr: Optional[str] = Field(
        default=None, description="Potential MRR if mentioned (e.g. '480 EUR')"
    )
    deal_stage: Optional[str] = Field(
        default=None, description="Deal stage if identifiable (e.g. 'Discovery', 'Negotiation')"
    )
