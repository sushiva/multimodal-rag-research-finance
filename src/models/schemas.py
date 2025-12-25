"""
Pydantic Models & Schemas

What: Type-safe request/response models for API
Why: Automatic validation, serialization, and API documentation
How: Pydantic BaseModel with Field validators

All API requests and responses use these schemas for:
- Input validation (reject invalid data early)
- Output serialization (consistent JSON format)
- OpenAPI documentation (auto-generated from schemas)
- Type safety (catches errors at development time)

Usage:
    from src.models.schemas import RAGQueryRequest, RAGQueryResponse

    request = RAGQueryRequest(query="What papers discuss transformers?", top_k=5)
    response = RAGQueryResponse(answer="...", sources=[...])
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """
    Supported document types.

    Values:
        ARXIV: Research papers from arXiv
        FINANCIAL: SEC financial filings
    """
    ARXIV = "arxiv"
    FINANCIAL = "financial"


class SearchMode(str, Enum):
    """
    Search mode types.

    Values:
        TEXT: Text-only search (BM25 + vector)
        VISUAL: Visual-only search (visual embeddings)
        MULTIMODAL: Combined text and visual search
    """
    TEXT = "text"
    VISUAL = "visual"
    MULTIMODAL = "multimodal"


class LLMProvider(str, Enum):
    """
    LLM provider types.

    Values:
        GEMINI: Google Gemini
        CLAUDE: Anthropic Claude
        OPENAI: OpenAI GPT
    """
    GEMINI = "gemini"
    CLAUDE = "claude"
    OPENAI = "openai"


class RAGQueryRequest(BaseModel):
    """
    Request schema for RAG query endpoint.

    Attributes:
        query: User question (required)
        document_type: Type of documents to search
        top_k: Number of results to retrieve
        ticker: Stock ticker filter (financial only)
        filing_types: Filing type filter (financial only)
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question or search query",
        examples=["What papers discuss transformer architectures?"]
    )

    document_type: DocumentType = Field(
        default=DocumentType.ARXIV,
        description="Type of documents to search"
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )

    ticker: Optional[str] = Field(
        default=None,
        max_length=10,
        description="Stock ticker symbol (e.g., AAPL, MSFT) for financial documents",
        examples=["AAPL", "MSFT", "GOOGL"]
    )

    filing_types: Optional[List[str]] = Field(
        default=None,
        description="SEC filing types to filter (e.g., 10-K, 10-Q)",
        examples=[["10-K"], ["10-K", "10-Q"]]
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query string."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        """Validate ticker symbol format."""
        if v:
            v = v.strip().upper()
            if not v.isalpha():
                raise ValueError("Ticker must contain only letters")
            if len(v) > 5:
                raise ValueError("Ticker too long (max 5 characters)")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are Apple's main business segments?",
                "document_type": "financial",
                "top_k": 3,
                "ticker": "AAPL",
                "filing_types": ["10-K"]
            }
        }


class RetrievalMetrics(BaseModel):
    """
    Metrics for retrieval process.

    Attributes:
        text_results: Number of text search results
        visual_results: Number of visual search results
        fusion_method: Method used for result fusion
    """

    text_results: int = Field(default=0, description="Number of text search results")
    visual_results: int = Field(default=0, description="Number of visual search results")
    fusion_method: Optional[str] = Field(
        default=None,
        description="Fusion method used (e.g., RRF, linear)"
    )


class TokenUsage(BaseModel):
    """
    LLM token usage metrics.

    Attributes:
        prompt_tokens: Tokens in prompt
        completion_tokens: Tokens in completion
        total_tokens: Total tokens used
    """

    prompt_tokens: int = Field(default=0, description="Tokens in prompt")
    completion_tokens: int = Field(default=0, description="Tokens in completion")
    total_tokens: int = Field(default=0, description="Total tokens used")


class RAGQueryResponse(BaseModel):
    """
    Response schema for RAG query endpoint.

    Attributes:
        answer: Generated answer text
        sources: List of source URLs
        chunks_used: Number of chunks used for generation
        search_mode: Search mode used
        retrieval_breakdown: Detailed retrieval metrics
        model_used: LLM model name
        provider_used: LLM provider
        tokens_used: Token usage metrics
        latency_ms: Total query latency in milliseconds
        confidence: Confidence score (low, medium, high)
        citations: List of citation strings
    """

    answer: str = Field(..., description="Generated answer text")

    sources: List[str] = Field(
        default_factory=list,
        description="List of source URLs"
    )

    chunks_used: int = Field(default=0, description="Number of chunks used")

    search_mode: SearchMode = Field(
        default=SearchMode.TEXT,
        description="Search mode used"
    )

    retrieval_breakdown: Optional[RetrievalMetrics] = Field(
        default=None,
        description="Detailed retrieval metrics"
    )

    model_used: Optional[str] = Field(
        default=None,
        description="LLM model name"
    )

    provider_used: Optional[LLMProvider] = Field(
        default=None,
        description="LLM provider used"
    )

    tokens_used: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage metrics"
    )

    latency_ms: Optional[int] = Field(
        default=None,
        ge=0,
        description="Total query latency in milliseconds"
    )

    confidence: str = Field(
        default="medium",
        description="Confidence score: low, medium, high"
    )

    citations: List[str] = Field(
        default_factory=list,
        description="List of citation strings"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Apple's main business segments include iPhone, Mac, iPad, Wearables (including Apple Watch and AirPods), and Services (including App Store, iCloud, Apple Music).",
                "sources": [
                    "https://www.sec.gov/cgi-bin/viewer?action=view&cik=320193&accession_number=0000320193-23-000077",
                ],
                "chunks_used": 3,
                "search_mode": "text",
                "model_used": "gemini-2.0-flash-exp",
                "provider_used": "gemini",
                "latency_ms": 2340,
                "confidence": "high",
                "citations": ["AAPL 10-K (2023)"]
            }
        }


class HealthStatus(str, Enum):
    """
    Health check status values.

    Values:
        HEALTHY: All services operational
        DEGRADED: Some services down but core functionality works
        UNHEALTHY: Critical services down
    """
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ServiceHealth(BaseModel):
    """
    Individual service health status.

    Attributes:
        status: Service status
        message: Optional status message
        latency_ms: Service response latency
    """

    status: HealthStatus = Field(..., description="Service health status")
    message: Optional[str] = Field(default=None, description="Status message")
    latency_ms: Optional[int] = Field(
        default=None,
        description="Service latency in milliseconds"
    )


class HealthCheckResponse(BaseModel):
    """
    Response schema for health check endpoint.

    Attributes:
        status: Overall system status
        timestamp: Check timestamp
        services: Individual service statuses
        version: Application version
    """

    status: HealthStatus = Field(..., description="Overall system status")

    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Health check timestamp"
    )

    services: Dict[str, ServiceHealth] = Field(
        default_factory=dict,
        description="Individual service health statuses"
    )

    version: str = Field(default="0.1.0", description="Application version")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2024-12-24T10:00:00Z",
                "services": {
                    "opensearch": {
                        "status": "healthy",
                        "latency_ms": 15
                    },
                    "postgres": {
                        "status": "healthy",
                        "latency_ms": 8
                    },
                    "redis": {
                        "status": "healthy",
                        "latency_ms": 2
                    }
                },
                "version": "0.1.0"
            }
        }


class ErrorResponse(BaseModel):
    """
    Standard error response schema.

    Attributes:
        error: Error type/code
        message: Human-readable error message
        details: Optional additional details
        request_id: Request ID for tracing
    """

    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracing"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "error": "validation_error",
                "message": "Query cannot be empty",
                "details": {
                    "field": "query",
                    "value": ""
                },
                "request_id": "req_abc123"
            }
        }
