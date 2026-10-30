"""AI Investigation Controller Package.

Coordinates the LLM, deterministic analytics engine, and investigation state model
to conduct an evidence-driven product investigation.
"""

from controller.capability_catalogue import (
    CONTROLLED_OPERATIONS,
    describe_capabilities,
    validate_analysis_plan,
)
from controller.controller import InvestigationController
from controller.llm_client import (
    GenericHTTPClient,
    LLMClient,
    LLMResponse,
    MockLLMClient,
    create_llm_client,
)
from controller.schemas import (
    AnalysisDecision,
    EvidenceInterpretation,
    HypothesesGenerationOutput,
    HypothesisProposal,
)

__all__ = [
    "InvestigationController",
    "LLMClient",
    "MockLLMClient",
    "GenericHTTPClient",
    "create_llm_client",
    "LLMResponse",
    "HypothesisProposal",
    "HypothesesGenerationOutput",
    "AnalysisDecision",
    "EvidenceInterpretation",
    "CONTROLLED_OPERATIONS",
    "validate_analysis_plan",
    "describe_capabilities",
]
