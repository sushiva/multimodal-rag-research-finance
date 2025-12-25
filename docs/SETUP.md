# Development Setup Guide

**Purpose:** Step-by-step instructions to set up local development environment.

**Target Audience:** Developers setting up the project for the first time.

---

## Prerequisites

### Required Software

1. **Python 3.11+**
   ```bash
   python --version
   # Should show: Python 3.11.x or higher
   ```

2. **Poetry** (or uv for faster dependency management)
   ```bash
   # Install Poetry
   curl -sSL https://install.python-poetry.org | python3 -

   # OR install uv (faster alternative)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Docker & Docker Compose** (for local services)
   ```bash
   docker --version
   docker-compose --version
   ```

4. **Git** (for version control)
   ```bash
   git --version
   ```

---

## Step 1: Clone Repository

```bash
# If you haven't cloned yet
git clone git@github.com:sushiva/multimodal-rag-research-finance.git
cd multimodal-rag-research-finance

# If already cloned, pull latest
git pull origin main
```

---

## Step 2: Set Up Python Environment

### Option A: Using Poetry (Standard)

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "import fastapi; print(fastapi.__version__)"
```

### Option B: Using uv (Faster)

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows

# Install dependencies
uv pip install -e ".[dev]"
```

### Option C: Using pip (If no Poetry/uv)

```bash
# Create virtual environment
python -m venv .venv

# Activate
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e ".[dev]"
```

**Why virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system Python
- Reproducible environment

---

## Step 3: Configure Environment Variables

```bash
# Copy example env file
cp .env.example .env

# Edit with your API keys
nano .env  # or use your preferred editor
```

**Required API Keys:**

1. **Google Gemini** (Primary LLM)
   - Get key: https://aistudio.google.com/app/apikey
   - Add to `.env`: `GEMINI_API_KEY=your_key`

2. **Anthropic Claude** (Fallback LLM)
   - Get key: https://console.anthropic.com/
   - Add to `.env`: `ANTHROPIC_API_KEY=your_key`

3. **OpenAI** (Last resort LLM)
   - Get key: https://platform.openai.com/api-keys
   - Add to `.env`: `OPENAI_API_KEY=your_key`

4. **Nomic** (Visual embeddings)
   - Get key: https://atlas.nomic.ai/
   - Add to `.env`: `NOMIC_API_KEY=your_key`

5. **Jina AI** (Text embeddings)
   - Get key: https://jina.ai/
   - Add to `.env`: `JINA_API_KEY=your_key`

**Generate SECRET_KEY:**
```bash
# Generate secure secret key
openssl rand -hex 32

# Add to .env
SECRET_KEY=<generated_key>
```

---

## Step 4: Start Infrastructure Services

### Start Docker Services

```bash
# Start OpenSearch, PostgreSQL, Redis
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

**Services started:**
- OpenSearch (port 9200): Vector search
- PostgreSQL (port 5432): SQL database
- Redis (port 6379): Cache

### Verify Services

```bash
# Test OpenSearch
curl http://localhost:9200

# Test PostgreSQL
psql postgresql://postgres:postgres@localhost:5432/multimodal_rag -c "SELECT 1;"

# Test Redis
redis-cli ping
# Should return: PONG
```

---

## Step 5: Initialize Database

```bash
# Run database migrations
alembic upgrade head

# Verify tables created
psql postgresql://postgres:postgres@localhost:5432/multimodal_rag -c "\dt"
```

---

## Step 6: Create OpenSearch Indices

```bash
# Run index creation script
python scripts/create_opensearch_indices.py

# Verify indices created
curl http://localhost:9200/_cat/indices
```

**Expected indices:**
- `arxiv` (text embeddings)
- `financial` (text embeddings)
- `arxiv_visual` (visual embeddings)
- `financial_visual` (visual embeddings)

---

## Step 7: Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run only unit tests
pytest -m unit

# Run specific test file
pytest tests/unit/services/test_vision.py
```

**Expected result:** All tests passing ✅

---

## Step 8: Start Development Server

```bash
# Start FastAPI server with auto-reload
uvicorn src.main:app --reload --port 8000

# OR using Python
python -m uvicorn src.main:app --reload --port 8000
```

**Access:**
- API: http://localhost:8000
- API Docs (Swagger): http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

## Step 9: Verify Setup

```bash
# Test health endpoint
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "services": {
#     "opensearch": "healthy",
#     "postgres": "healthy",
#     "redis": "healthy"
#   }
# }
```

---

## Development Workflow

### Daily Development

```bash
# 1. Activate environment
poetry shell  # or source .venv/bin/activate

# 2. Pull latest changes
git pull origin main

# 3. Install new dependencies (if any)
poetry install

# 4. Start services
docker-compose up -d

# 5. Start dev server
uvicorn src.main:app --reload
```

### Before Committing

```bash
# 1. Format code
black src/ tests/
isort src/ tests/

# 2. Lint
ruff check src/ tests/

# 3. Type check
mypy src/

# 4. Run tests
pytest

# 5. Commit
git add .
git commit -m "Your message"
git push origin your-branch
```

---

## Troubleshooting

### Issue: `poetry install` fails

**Solution:**
```bash
# Clear cache
poetry cache clear pypi --all

# Remove lock file and retry
rm poetry.lock
poetry install
```

### Issue: Docker services won't start

**Solution:**
```bash
# Check if ports are in use
lsof -i :9200  # OpenSearch
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis

# Stop conflicting services
docker-compose down
docker-compose up -d
```

### Issue: OpenSearch health check fails

**Solution:**
```bash
# Wait for OpenSearch to initialize (30-60 seconds)
docker-compose logs opensearch

# Check if running
curl http://localhost:9200/_cluster/health
```

### Issue: Database migrations fail

**Solution:**
```bash
# Drop and recreate database
dropdb multimodal_rag
createdb multimodal_rag

# Run migrations again
alembic upgrade head
```

### Issue: Import errors

**Solution:**
```bash
# Ensure you're in virtual environment
which python
# Should show: /path/to/project/.venv/bin/python

# Reinstall in editable mode
pip install -e .
```

---

## IDE Setup

### VS Code

**Recommended Extensions:**
- Python (Microsoft)
- Pylance
- Python Test Explorer
- Docker
- REST Client

**Settings (.vscode/settings.json):**
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
```

### PyCharm

1. Open project directory
2. PyCharm detects `pyproject.toml` automatically
3. Select Python interpreter: `.venv/bin/python`
4. Enable "Black" formatter in settings

---

## Next Steps

After setup is complete:

1. ✅ Read `docs/ARCHITECTURE.md` (understand system design)
2. ✅ Read `docs/MULTIMODAL_RAG_PLAN.md` (implementation roadmap)
3. ✅ Start implementing core services (see roadmap)

---

## Resources

### Documentation
- FastAPI: https://fastapi.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- OpenSearch: https://opensearch.org/docs/
- SQLAlchemy: https://docs.sqlalchemy.org/

### Tools
- Poetry: https://python-poetry.org/docs/
- Docker Compose: https://docs.docker.com/compose/
- pytest: https://docs.pytest.org/

---

**Last Updated:** December 2024
**Maintainer:** Sudhir Shivaram
**Status:** Living Document
