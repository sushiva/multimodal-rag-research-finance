# Frequently Asked Questions (FAQ)

**Purpose:** Common questions about architecture, design decisions, and implementation choices.

**Target Audience:** Developers working on or evaluating this project.

---

## Table of Contents

1. [Why pyproject.toml instead of requirements.txt?](#why-pyprojecttoml-instead-of-requirementstxt)
2. [What is Pydantic's role in this implementation?](#what-is-pydantics-role-in-this-implementation)

---

## Why pyproject.toml instead of requirements.txt?

**Short Answer:** pyproject.toml is the modern Python standard (PEP 518) that provides dependency resolution, version management, and tool configuration in a single file. requirements.txt is the legacy approach requiring multiple files with manual conflict resolution.

### Key Advantages

**1. Single Source of Truth**
- Project metadata, dependencies, dev dependencies, and tool configurations all in one file
- With requirements.txt, you need: `requirements.txt`, `requirements-dev.txt`, `setup.py`, `pytest.ini`, etc.

**2. Automatic Dependency Resolution**
```bash
# pyproject.toml with Poetry
poetry install  # Resolves entire dependency tree, detects conflicts

# requirements.txt with pip
pip install -r requirements.txt  # No conflict detection, picks arbitrary versions
```

**3. Smart Version Management (Semantic Versioning)**
```toml
# pyproject.toml - Flexible but safe
fastapi = "^0.104.1"  # Allows 0.104.1 to <1.0.0 (patch/minor updates only)

# requirements.txt - Manual and error-prone
fastapi==0.104.1  # Too strict, blocks security updates
# OR
fastapi>=0.104.1  # Too loose, could break with 1.0.0
```

**4. Separate Production and Development Dependencies**
```toml
[tool.poetry.dependencies]
fastapi = "^0.104.1"  # Production only

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.3"  # Development only, excluded from production builds
```

```bash
poetry install --only main  # Production (no pytest, black, mypy)
poetry install --with dev   # Development (everything)
```

**5. Reproducible Builds with Lockfile**
- `pyproject.toml` defines what you want
- `poetry.lock` defines exactly what you get (locks every sub-dependency version)
- Everyone on the team gets identical environments

**6. Official Python Standard**
- PEP 518 (2016) standardized `pyproject.toml` as the official Python packaging format
- All modern tools (Poetry, Hatch, PDM) use it
- requirements.txt is the legacy approach

### Tools Configured in pyproject.toml

Our project configures everything in one place:

```toml
[tool.black]           # Code formatter
line-length = 100

[tool.ruff]            # Linter
line-length = 100

[tool.mypy]            # Type checker
python_version = "3.11"

[tool.pytest.ini_options]  # Test runner
minversion = "7.0"

[tool.coverage.run]    # Coverage tracking
source = ["src"]
```

Without pyproject.toml, you'd need separate config files for each tool.

### When requirements.txt is Still Used

1. **Legacy projects** already using it (migration effort not worth it)
2. **Simple single-file scripts** (no package structure)
3. **Docker builds** (though Poetry supports Docker with multi-stage builds)

### Our Project Setup

See [pyproject.toml](../pyproject.toml) for complete configuration.

**One-command setup:**
```bash
poetry install  # Installs all dependencies, resolves conflicts, activates environment
poetry shell    # Activate virtual environment
```

**Equivalent with requirements.txt (messier):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
# Configure black, pytest, mypy separately
# Hope there are no dependency conflicts
```

---

## What is Pydantic's role in this implementation?

**Short Answer:** Pydantic provides type-safe data validation, configuration management, and automatic API documentation for FastAPI. It's the backbone of our request/response handling, environment configuration, and data integrity.

### Three Core Uses in Our System

#### 1. API Request/Response Validation

**Location:** [src/models/schemas.py](../src/models/schemas.py)

Pydantic models define all API contracts with automatic validation:

```python
from pydantic import BaseModel, Field, field_validator

class RAGQueryRequest(BaseModel):
    """Request schema for RAG query endpoint."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="User question or search query"
    )

    top_k: int = Field(default=5, ge=1, le=20)
    ticker: Optional[str] = Field(default=None, max_length=10)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query string."""
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty")
        return v
```

**What Pydantic Does:**
- ✅ Validates `query` is 3-1000 characters (rejects invalid requests before they reach business logic)
- ✅ Validates `top_k` is between 1-20 (prevents resource abuse)
- ✅ Auto-converts types (e.g., `"5"` → `5` for integers)
- ✅ Custom validation with `@field_validator` (sanitizes input)
- ✅ Generates OpenAPI schema for automatic API documentation

**FastAPI Integration:**
```python
from fastapi import FastAPI
from src.models.schemas import RAGQueryRequest, RAGQueryResponse

app = FastAPI()

@app.post("/query", response_model=RAGQueryResponse)
async def query_documents(request: RAGQueryRequest):
    # FastAPI automatically:
    # 1. Parses JSON body
    # 2. Validates against RAGQueryRequest schema
    # 3. Returns 422 error if validation fails
    # 4. Converts response to RAGQueryResponse format

    # request.query is guaranteed to be valid here
    return {"answer": "...", "sources": [...]}
```

#### 2. Configuration Management

**Location:** [src/core/config.py](../src/core/config.py)

Pydantic Settings loads and validates environment variables:

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
    )

    # LLM API Keys (required)
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Database Configuration
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432, ge=1, le=65535)

    # Redis Cache Settings
    cache_embedding_ttl: int = Field(default=604800, ge=0)  # 7 days

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()
```

**What Pydantic Settings Does:**
- ✅ Loads from `.env` file automatically
- ✅ Validates all values at startup (fails fast if config is wrong)
- ✅ Type conversion (`"5432"` string → `5432` integer)
- ✅ Validates constraints (`postgres_port` must be 1-65535)
- ✅ Provides defaults for optional settings
- ✅ Case-insensitive matching (`POSTGRES_HOST` or `postgres_host` both work)

**Usage in Application:**
```python
from src.core.config import get_settings

settings = get_settings()  # Load once, cached forever

# Type-safe access (IDE autocomplete, mypy validation)
db_url = f"postgresql://{settings.postgres_host}:{settings.postgres_port}"

# Guaranteed to be valid (validated at startup)
redis_ttl = settings.cache_embedding_ttl  # Always a positive integer
```

#### 3. Data Integrity and Type Safety

Pydantic ensures type safety throughout the codebase:

```python
from typing import List
from pydantic import BaseModel

class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class RAGQueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    tokens_used: Optional[TokenUsage] = None

# Type-safe construction
response = RAGQueryResponse(
    answer="Apple's revenue grew 10%",
    sources=["https://sec.gov/..."],
    tokens_used=TokenUsage(prompt_tokens=150, completion_tokens=80)
)

# Pydantic validates:
# - answer is a string (not None, not int)
# - sources is a list of strings (not a single string)
# - tokens_used.prompt_tokens is an integer (not "150" string)

# Serialization to JSON (for API responses)
json_data = response.model_dump()  # Python dict
json_str = response.model_dump_json()  # JSON string
```

### Why Pydantic Over Alternatives?

**vs. dataclasses (standard library):**
- ❌ dataclasses: No validation, no environment loading, no JSON serialization
- ✅ Pydantic: Full validation, automatic coercion, JSON support

**vs. marshmallow (older validation library):**
- ❌ marshmallow: Separate schema and data objects, verbose
- ✅ Pydantic: Single object is both data and schema, cleaner

**vs. attrs (validation library):**
- ❌ attrs: Not designed for API validation, no FastAPI integration
- ✅ Pydantic: Built for FastAPI, automatic OpenAPI docs

### Performance Considerations

Pydantic V2 (what we're using) is **extremely fast**:
- Written in Rust (pydantic-core)
- 5-50x faster than Pydantic V1
- Benchmarks show it's faster than manual validation in most cases

```toml
# pyproject.toml
pydantic = "^2.5.0"  # Version 2.x (Rust-powered)
```

### Key Files Using Pydantic

1. **[src/models/schemas.py](../src/models/schemas.py)** - All API models
   - `RAGQueryRequest`, `RAGQueryResponse`
   - `HealthCheckResponse`, `ErrorResponse`
   - Enums: `DocumentType`, `SearchMode`, `LLMProvider`

2. **[src/core/config.py](../src/core/config.py)** - Configuration management
   - `Settings` class (Pydantic Settings)
   - Environment variable loading and validation

3. **Future service modules** (not yet implemented)
   - Database models (SQLAlchemy + Pydantic)
   - Internal data transfer objects

### Pydantic in the Request Flow

```
1. HTTP Request → FastAPI
   ↓
2. JSON body → Pydantic validation (RAGQueryRequest)
   ↓
3. If valid → Business logic (query processing)
   ↓
4. Business logic → Response data
   ↓
5. Response data → Pydantic serialization (RAGQueryResponse)
   ↓
6. JSON response → Client
```

**Error handling:**
```python
# Invalid request example
POST /query
{
  "query": "ab",  # Too short (min 3 chars)
  "top_k": 100    # Too large (max 20)
}

# Pydantic automatically returns:
HTTP 422 Unprocessable Entity
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "String should have at least 3 characters",
      "type": "string_too_short"
    },
    {
      "loc": ["body", "top_k"],
      "msg": "Input should be less than or equal to 20",
      "type": "less_than_equal"
    }
  ]
}
```

### Summary

**Pydantic's role in our system:**

1. **Data Validation** - Rejects invalid data before it reaches business logic
2. **Type Safety** - Catches errors at development time with mypy
3. **Configuration Management** - Loads and validates environment variables
4. **API Documentation** - Auto-generates OpenAPI schemas for FastAPI
5. **Serialization** - Converts between Python objects and JSON
6. **Developer Experience** - IDE autocomplete, clear error messages

**Without Pydantic, we'd need:**
- Manual validation for every request field
- Manual type conversions (string → int, etc.)
- Manual environment variable loading
- Manual OpenAPI schema writing
- Custom serialization logic
- More runtime errors from invalid data

Pydantic is the foundation that makes our FastAPI application type-safe, self-documenting, and robust.

---

**Last Updated:** December 2024
**Maintainer:** Sudhir Shivaram
**Status:** Living Document
