"""
Application Configuration

What: Centralized configuration management using Pydantic Settings
Why: Type-safe settings, automatic validation, environment variable loading
How: Pydantic BaseSettings loads from .env file and validates types

Usage:
    from src.core.config import get_settings
    settings = get_settings()
    print(settings.gemini_api_key)
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings are validated at startup. Missing required values
    will raise ValidationError with clear error messages.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(
        default="multimodal-rag-research-finance",
        description="Application name"
    )
    environment: str = Field(
        default="development",
        description="Environment: development, staging, production"
    )
    debug: bool = Field(
        default=False,
        description="Debug mode (verbose logging, auto-reload)"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    )

    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_reload: bool = Field(default=False, description="Auto-reload on code changes")

    secret_key: str = Field(
        ...,
        description="Secret key for JWT signing (generate with: openssl rand -hex 32)"
    )
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8501"],
        description="CORS allowed origins"
    )

    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp",
        description="Gemini model name"
    )
    gemini_max_tokens: int = Field(default=2000, description="Max tokens for Gemini")

    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    anthropic_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Claude model name"
    )
    anthropic_max_tokens: int = Field(default=2000, description="Max tokens for Claude")

    openai_api_key: str = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    openai_max_tokens: int = Field(default=500, description="Max tokens for OpenAI")

    jina_api_key: str = Field(default="", description="Jina AI API key")
    jina_model: str = Field(
        default="jina-embeddings-v3",
        description="Jina embedding model"
    )
    jina_dimensions: int = Field(default=1024, description="Jina embedding dimensions")

    nomic_api_key: str = Field(default="", description="Nomic API key")
    nomic_model: str = Field(
        default="nomic-embed-vision-v1.5",
        description="Nomic visual embedding model"
    )
    nomic_dimensions: int = Field(default=768, description="Nomic embedding dimensions")

    opensearch_url: str = Field(
        default="http://localhost:9200",
        description="OpenSearch URL"
    )
    opensearch_username: str = Field(default="admin", description="OpenSearch username")
    opensearch_password: str = Field(default="admin", description="OpenSearch password")
    opensearch_use_ssl: bool = Field(default=False, description="Use SSL for OpenSearch")
    opensearch_verify_certs: bool = Field(
        default=False,
        description="Verify SSL certificates"
    )

    opensearch_index_arxiv: str = Field(default="arxiv", description="arXiv index name")
    opensearch_index_financial: str = Field(
        default="financial",
        description="Financial index name"
    )
    opensearch_index_arxiv_visual: str = Field(
        default="arxiv_visual",
        description="arXiv visual index name"
    )
    opensearch_index_financial_visual: str = Field(
        default="financial_visual",
        description="Financial visual index name"
    )

    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="multimodal_rag", description="PostgreSQL database")
    postgres_user: str = Field(default="postgres", description="PostgreSQL username")
    postgres_password: str = Field(default="postgres", description="PostgreSQL password")
    database_url: Optional[str] = Field(
        default=None,
        description="Full PostgreSQL URL (overrides individual fields)"
    )

    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database number")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_url: Optional[str] = Field(
        default=None,
        description="Full Redis URL (overrides individual fields)"
    )

    cache_embedding_ttl: int = Field(
        default=604800,
        description="Embedding cache TTL (seconds, default 7 days)"
    )
    cache_response_ttl: int = Field(
        default=3600,
        description="Response cache TTL (seconds, default 1 hour)"
    )
    cache_default_ttl: int = Field(
        default=300,
        description="Default cache TTL (seconds, default 5 minutes)"
    )

    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_per_minute: int = Field(
        default=60,
        description="Requests per minute per user"
    )
    rate_limit_per_hour: int = Field(
        default=1000,
        description="Requests per hour per user"
    )

    max_upload_size_mb: int = Field(
        default=10,
        description="Maximum file upload size in MB"
    )
    allowed_image_types: List[str] = Field(
        default=["image/png", "image/jpeg", "image/jpg"],
        description="Allowed image MIME types"
    )
    temp_upload_dir: str = Field(
        default="/tmp/uploads",
        description="Temporary upload directory"
    )

    sentry_dsn: Optional[str] = Field(default=None, description="Sentry DSN for error tracking")
    structured_logging: bool = Field(
        default=True,
        description="Use structured (JSON) logging"
    )
    log_format: str = Field(
        default="json",
        description="Log format: json or text"
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = ["development", "staging", "production"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"Log level must be one of: {allowed}")
        return v.upper()

    @field_validator("allowed_origins")
    @classmethod
    def parse_allowed_origins(cls, v: str | List[str]) -> List[str]:
        """Parse comma-separated string or list of origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    def get_database_url(self) -> str:
        """
        Get PostgreSQL database URL.

        Returns full URL if provided, otherwise constructs from components.
        """
        if self.database_url:
            return self.database_url
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def get_redis_url(self) -> str:
        """
        Get Redis URL.

        Returns full URL if provided, otherwise constructs from components.
        """
        if self.redis_url:
            return self.redis_url

        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Why cache: Settings are immutable and expensive to load.
    Using lru_cache ensures we only load once per process.

    Returns:
        Settings instance loaded from environment

    Raises:
        ValidationError: If required settings are missing or invalid
    """
    return Settings()
