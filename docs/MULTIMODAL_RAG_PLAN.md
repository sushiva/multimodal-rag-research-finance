# Multimodal RAG System - Comprehensive Implementation Plan

**Project:** Multimodal Research & Financial Document RAG
**Repository:** `multimodal-rag-research-finance`
**Status:** Planning Phase
**Start Date:** TBD
**Estimated Timeline:** 3-4 weeks (part-time)

---

## 📋 Table of Contents

1. [Executive Summary](#executive-summary)
2. [Project Vision](#project-vision)
3. [Architecture Overview](#architecture-overview)
4. [Implementation Strategy](#implementation-strategy)
5. [Cost Analysis](#cost-analysis)
6. [Technical Specifications](#technical-specifications)
7. [Deployment Plan](#deployment-plan)
8. [Portfolio Positioning](#portfolio-positioning)
9. [Risk Mitigation](#risk-mitigation)
10. [Success Metrics](#success-metrics)

---

## 📊 Executive Summary

### **What We're Building**
A cutting-edge **multimodal RAG system** that enables visual similarity search across research papers and financial documents. Users can query by:
- Text descriptions (existing capability)
- Visual patterns (diagrams, charts, tables)
- Combined text + visual queries

### **Why This Matters**
- **For Researchers**: Search by transformer diagrams, architecture visualizations, chart patterns
- **For Financial Analysts**: Find similar balance sheets, revenue breakdowns, org charts
- **For Your Portfolio**: Demonstrates advanced AI/ML skills beyond typical RAG demos

### **Key Differentiators**
1. **Dual document types** (Research + Financial) with different visual patterns
2. **From-scratch implementation** (no LangChain overhead) showing first-principles understanding
3. **Production deployment** on Railway with full monitoring
4. **Cost-optimized** (+$8-12/month over v1, still under $25/month total)

### **Strategic Decision: New Project vs Evolving Current**
**Decision: Create new project** (`multimodal-rag-research-finance`)

**Rationale:**
- ✅ Portfolio variety (2 projects instead of 1)
- ✅ Better storytelling (progression from basic → advanced)
- ✅ Risk management (v1 stays production-stable)
- ✅ GitHub visibility (2 repos = more activity)
- ✅ Interview narrative (shows innovation + iteration)

---

## 🎯 Project Vision

### **Current State (v1): Text-Based RAG**
```
User Query (Text) → Hybrid Search (BM25 + Vector) → LLM → Answer
                     └── OpenSearch: Text embeddings (1024-dim)
```

**What it does well:**
- Fast text retrieval (2-3 seconds)
- 4-tier LLM fallback (99.9% uptime)
- Dual index (arXiv + Financial)
- Production-ready

**What it cannot do:**
- Search by visual similarity
- Find papers with similar diagrams
- Retrieve documents by table/chart structure

### **Future State (v2): Multimodal RAG**
```
User Query (Text + Optional Image) → Multimodal Search → LLM → Answer
                                      ├── Text embeddings (Jina 1024-dim)
                                      └── Visual embeddings (Nomic 768-dim)
                                          ↓
                                    Fusion Ranking (RRF)
```

**New capabilities:**
- ✅ Search papers by diagram similarity
- ✅ Find financial docs by table structure
- ✅ Query by chart patterns (revenue trends, loss curves)
- ✅ Combined text + visual retrieval

### **Example Queries**

#### **arXiv Papers (Research)**
| Query Type | Example |
|------------|---------|
| Text only | "Papers discussing transformer architectures" |
| Visual only | [Image of transformer diagram] → "Find similar architectures" |
| Hybrid | "Papers about GANs with generator-discriminator diagrams" |

#### **Financial Documents**
| Query Type | Example |
|------------|---------|
| Text only | "Companies discussing AI revenue growth" |
| Visual only | [Image of revenue breakdown] → "Find similar revenue structures" |
| Hybrid | "Tech companies with similar quarterly trends to NVDA" |

---

## 🏗️ Architecture Overview

### **High-Level System Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Frontend                       │
│  - Text query input                                          │
│  - Image upload (optional)                                   │
│  - Document type selector (arXiv / Financial)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Railway)                  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         Multimodal Retrieval Service                 │   │
│  │                                                       │   │
│  │  ┌──────────────┐         ┌──────────────┐         │   │
│  │  │ Text Search  │         │ Visual Search │         │   │
│  │  │              │         │               │         │   │
│  │  │ BM25 + Jina  │         │ Nomic Vision  │         │   │
│  │  │ (1024-dim)   │         │ (768-dim)     │         │   │
│  │  └──────┬───────┘         └───────┬───────┘         │   │
│  │         │                         │                 │   │
│  │         └──────────┬──────────────┘                 │   │
│  │                    ▼                                 │   │
│  │          Fusion Ranking (RRF)                        │   │
│  │                    │                                 │   │
│  └────────────────────┼─────────────────────────────────┘   │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │         4-Tier LLM Fallback                          │   │
│  │  Gemini Flash → Gemini Pro → Claude → OpenAI        │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Layer (Railway)                       │
│                                                               │
│  ┌──────────────┐    ┌──────────────────────────────────┐  │
│  │ PostgreSQL   │    │      OpenSearch Indices          │  │
│  │              │    │                                   │  │
│  │ - Metadata   │    │  ┌────────────┐  ┌─────────────┐ │  │
│  │ - Papers     │    │  │ arxiv      │  │ financial   │ │  │
│  │ - Filings    │    │  │ (text)     │  │ (text)      │ │  │
│  └──────────────┘    │  └────────────┘  └─────────────┘ │  │
│                      │                                   │  │
│  ┌──────────────┐    │  ┌────────────┐  ┌─────────────┐ │  │
│  │ Redis Cache  │    │  │ arxiv      │  │ financial   │ │  │
│  │              │    │  │ _visual    │  │ _visual     │ │  │
│  │ - Embeddings │    │  │ (768-dim)  │  │ (768-dim)   │ │  │
│  │ - Responses  │    │  └────────────┘  └─────────────┘ │  │
│  └──────────────┘    └──────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                      ▲
                      │
┌─────────────────────┴───────────────────────────────────────┐
│              Ingestion Pipeline (Airflow)                    │
│                                                               │
│  PDF → Image Extraction → Visual Embeddings → OpenSearch    │
│         (PyMuPDF)          (Nomic Vision)      (Indexing)    │
└─────────────────────────────────────────────────────────────┘
```

### **Data Indices**

| Index Name | Purpose | Embedding Type | Dimensions |
|------------|---------|----------------|------------|
| `arxiv` | Text search for papers | Jina AI v3 | 1024 |
| `financial` | Text search for filings | Jina AI v3 | 1024 |
| `arxiv_visual` 🆕 | Visual search for papers | Nomic Vision v1.5 | 768 |
| `financial_visual` 🆕 | Visual search for filings | Nomic Vision v1.5 | 768 |

### **Service Architecture**

```python
src/
├── services/
│   ├── vision/              # 🆕 NEW - Visual processing
│   │   ├── __init__.py
│   │   ├── pdf_extractor.py      # Extract images from PDFs
│   │   ├── nomic_client.py       # Nomic Embed Vision API
│   │   └── image_processor.py    # Resize, normalize
│   │
│   ├── search/
│   │   ├── opensearch.py         # ✅ Existing
│   │   └── multimodal_retriever.py  # 🆕 NEW - Dual retrieval
│   │
│   ├── gemini/              # ✅ Existing
│   ├── anthropic/           # ✅ Existing
│   └── openai/              # ✅ Existing
│
├── routers/
│   └── ask.py               # 📝 Updated - Add visual query param
│
└── ingestion/
    └── visual_embeddings.py  # 🆕 NEW - Batch visual processing
```

---

## 🚀 Implementation Strategy

### **Development Approach: From Scratch (No LangChain)**

**Rationale:**
1. ✅ Consistency with v1 architecture (clean service layers)
2. ✅ Better performance (no abstraction overhead)
3. ✅ Easier debugging (you control every line)
4. ✅ Portfolio value (shows first-principles understanding)
5. ✅ Lighter deployment (fewer dependencies)

**What we'll build:**
- Custom visual embedding service
- Custom multimodal retrieval logic
- Custom fusion ranking (extend existing RRF)

**What we'll use from ecosystem:**
- Nomic Embed Vision SDK (for embeddings)
- PyMuPDF (for PDF processing)
- Pillow (for image handling)

### **Phase-by-Phase Implementation**

#### **Phase 0: Project Setup (Week 0)**

**Goals:**
- Fork v1 to new repository
- Set up new Railway project
- Create project structure

**Tasks:**
1. Create new GitHub repo: `multimodal-rag-research-finance`
2. Fork code from v1
3. Update README with multimodal focus
4. Set up Railway project (separate from v1)
5. Configure environment variables
6. Test basic deployment (v1 features working)

**Deliverables:**
- ✅ New repo on GitHub
- ✅ Railway deployment (text-only, v1 features)
- ✅ Updated README
- ✅ CI/CD pipeline

**Time:** 1-2 days

---

#### **Phase 1: Visual Extraction & Embedding (Week 1)**

**Goals:**
- Extract images from PDFs
- Generate visual embeddings
- Test locally with FAISS

**Tasks:**

**Day 1-2: PDF Image Extraction**
```python
# Implement PDFImageExtractor
from pdf_extractor import PDFImageExtractor

extractor = PDFImageExtractor()
images = extractor.extract_pages_as_images(
    pdf_path="papers/attention_is_all_you_need.pdf",
    output_dir="temp/images/",
    dpi=150
)
# Output: ["temp/images/page_0.png", "temp/images/page_1.png", ...]
```

**Day 3-4: Visual Embeddings**
```python
# Implement NomicVisionClient
from nomic_client import NomicVisionClient

client = NomicVisionClient()
embeddings = await client.embed_images(images)
# Output: List[List[float]] - 768-dim vectors
```

**Day 5: Local Testing**
```python
# Test with FAISS locally
import faiss
import numpy as np

# Create index
embeddings_array = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(768)
index.add(embeddings_array)

# Query
query_image = "test_diagram.png"
query_embedding = await client.embed_images([query_image])
distances, indices = index.search(np.array(query_embedding), k=5)
```

**Test Dataset:** 20-30 arXiv papers (from existing corpus)

**Deliverables:**
- ✅ `PDFImageExtractor` class
- ✅ `NomicVisionClient` class
- ✅ Local FAISS tests passing
- ✅ Visual embeddings generated for test papers

**Time:** 5-6 days

---

#### **Phase 2: OpenSearch Visual Indices (Week 2)**

**Goals:**
- Create visual indices in OpenSearch
- Index visual embeddings
- Test vector similarity search

**Tasks:**

**Day 1-2: OpenSearch Index Creation**
```python
# Create arxiv_visual and financial_visual indices
from opensearch_service import OpenSearchService

os_client = OpenSearchService()

# Create visual index
os_client.create_index(
    index_name="arxiv_visual",
    dimensions=768,
    settings={
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "knn": True,
        "knn.algo_param.ef_search": 100
    }
)
```

**Day 3-4: Batch Indexing**
```python
# Index visual embeddings
for paper_id, page_embeddings in visual_data.items():
    for page_num, embedding in enumerate(page_embeddings):
        os_client.index_document(
            index="arxiv_visual",
            doc={
                "paper_id": paper_id,
                "page_number": page_num,
                "visual_embedding": embedding,
                "metadata": {...}
            }
        )
```

**Day 5: Test Visual Search**
```python
# Query visual index
query_embedding = [0.1, 0.2, ...]  # 768-dim
results = os_client.vector_search(
    index="arxiv_visual",
    embedding=query_embedding,
    top_k=10
)
```

**Deliverables:**
- ✅ `arxiv_visual` index created
- ✅ 20-30 papers indexed with visual embeddings
- ✅ Vector search working
- ✅ Performance benchmarks (<500ms query time)

**Time:** 5-6 days

---

#### **Phase 3: Multimodal Retrieval Logic (Week 3)**

**Goals:**
- Implement dual retrieval (text + visual)
- Add fusion ranking
- Update API endpoint

**Tasks:**

**Day 1-3: Multimodal Retriever**
```python
# src/services/search/multimodal_retriever.py
class MultimodalRetriever:
    async def retrieve(
        self,
        text_query: str,
        image_query: str = None,
        document_type: str = "arxiv",
        top_k: int = 5
    ):
        # Text retrieval (existing)
        text_results = await self.text_search(text_query, top_k)

        # Visual retrieval (new)
        if image_query:
            visual_results = await self.visual_search(image_query, top_k)

            # Fusion ranking (RRF)
            return self.reciprocal_rank_fusion(
                text_results,
                visual_results
            )

        return text_results
```

**Day 4-5: API Integration**
```python
# src/routers/ask.py - Update endpoint
@router.post("/ask")
async def ask_question(
    query: str,
    image: UploadFile = None,  # 🆕 NEW
    document_type: str = "arxiv",
    top_k: int = 5
):
    # Process image if provided
    image_path = None
    if image:
        image_path = await save_uploaded_image(image)

    # Multimodal retrieval
    retriever = MultimodalRetriever()
    results = await retriever.retrieve(
        text_query=query,
        image_query=image_path,
        document_type=document_type,
        top_k=top_k
    )

    # LLM generation (existing 4-tier fallback)
    answer = await llm_client.generate_rag_answer(query, results)
    return answer
```

**Deliverables:**
- ✅ `MultimodalRetriever` class
- ✅ Updated `/ask` endpoint with image upload
- ✅ Fusion ranking working
- ✅ End-to-end tests passing

**Time:** 5-6 days

---

#### **Phase 4: Airflow Integration (Week 4)**

**Goals:**
- Automate visual embedding generation
- Schedule batch processing
- Monitor pipeline health

**Tasks:**

**Day 1-3: Airflow DAG**
```python
# airflow/dags/visual_embedding_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator

def extract_images_from_pdfs(**context):
    """Extract images from newly added PDFs."""
    # Implementation

def generate_visual_embeddings(**context):
    """Generate embeddings using Nomic Vision."""
    # Implementation

def index_to_opensearch(**context):
    """Index visual embeddings to OpenSearch."""
    # Implementation

with DAG('visual_embedding_pipeline', schedule_interval='@daily'):
    extract = PythonOperator(task_id='extract_images', ...)
    embed = PythonOperator(task_id='generate_embeddings', ...)
    index = PythonOperator(task_id='index_embeddings', ...)

    extract >> embed >> index
```

**Day 4-5: Testing & Monitoring**
- Test DAG with sample papers
- Add error handling and retries
- Set up Airflow monitoring

**Deliverables:**
- ✅ Airflow DAG for visual processing
- ✅ Scheduled daily runs
- ✅ Error handling and monitoring
- ✅ Documentation

**Time:** 5 days

---

#### **Phase 5: Financial Documents (Week 4-5)**

**Goals:**
- Extend to financial documents
- Handle tables and charts
- Test financial visual queries

**Tasks:**

**Day 1-2: Financial PDF Processing**
```python
# Handle SEC filings (different structure than arXiv)
# Extract tables, charts separately
```

**Day 3-4: Financial Visual Index**
```python
# Create financial_visual index
# Index 10-20 financial documents
```

**Day 5: Testing**
- Test financial visual queries
- Validate table/chart similarity search

**Deliverables:**
- ✅ `financial_visual` index
- ✅ 10-20 financial docs indexed
- ✅ Visual search for financial docs working

**Time:** 5 days (can be done in parallel with Phase 4)

---

#### **Phase 6: UI Updates (Week 5)**

**Goals:**
- Add image upload to Streamlit
- Add "Search by image" feature
- Update UI/UX

**Tasks:**

**Day 1-2: Streamlit Image Upload**
```python
# streamlit_app.py
uploaded_image = st.file_uploader(
    "Upload an image to search by visual similarity (optional)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_image:
    st.image(uploaded_image, caption="Query Image", width=300)
```

**Day 3: API Integration**
```python
# Call /ask endpoint with image
files = {"image": uploaded_image} if uploaded_image else None
response = requests.post(
    f"{API_URL}/api/v1/ask",
    json={"query": query, "document_type": doc_type},
    files=files
)
```

**Day 4-5: Testing & Polish**
- Test image upload flow
- Add loading states
- Take screenshots for portfolio

**Deliverables:**
- ✅ Image upload UI
- ✅ Visual search working in Streamlit
- ✅ Screenshots for portfolio
- ✅ Updated user documentation

**Time:** 5 days

---

## 💰 Cost Analysis

### **Detailed Cost Breakdown**

#### **One-Time Setup Costs**

| Item | Cost | Notes |
|------|------|-------|
| PDF image extraction | $0 | PyMuPDF (free, local) |
| Visual embedding generation | $0-5 | Nomic free tier / Google Colab |
| Index creation | $0 | Part of OpenSearch |
| Testing & validation | $0 | Local development |
| **Total One-Time** | **$0-5** | |

#### **Monthly Operational Costs**

**V1 (Current Production System):**
| Component | Current Cost |
|-----------|--------------|
| Railway (FastAPI + OpenSearch + PostgreSQL) | $12/month |
| Streamlit Cloud | FREE |
| Jina AI embeddings | FREE (within limits) |
| LLM costs (4-tier fallback) | Included in Railway |
| **V1 Total** | **$12/month** |

**V2 (Multimodal System):**
| Component | Estimated Cost | Notes |
|-----------|----------------|-------|
| Railway Compute (FastAPI) | $8/month | Same as v1 |
| OpenSearch (4 indices) | $8-10/month | +2 visual indices |
| PostgreSQL | $3/month | Same as v1 |
| Redis Cache | $3-4/month | For embedding cache |
| Jina AI (text embeddings) | FREE | Within free tier |
| Nomic Vision (visual embeddings) | FREE | 1M tokens/month free |
| LLM costs | Included | Same 4-tier fallback |
| Streamlit Cloud | FREE | Same as v1 |
| **V2 Total** | **$22-25/month** | |

**Combined Total (Both Systems Running):**
- V1: $12/month
- V2: $22-25/month
- **Grand Total: $34-37/month**

**Cost Optimization Strategies:**
1. Use Redis to cache visual embeddings (reduce Nomic API calls)
2. Batch process images during off-peak hours
3. Use Railway free tier for dev/testing
4. Deploy v2 only when actively job hunting
5. Scale down OpenSearch replicas (can use 0 replicas for demo)

**Cost per Query:**
- Text query: ~$0.001
- Visual query: ~$0.002 (slightly higher due to larger index)
- Still incredibly cost-effective!

---

## 🛠️ Technical Specifications

### **Technology Stack**

#### **Backend**
| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| API Framework | FastAPI | 0.104+ | REST API |
| Vector Database | OpenSearch | 2.11+ | Text + visual embeddings |
| SQL Database | PostgreSQL | 15+ | Metadata, papers, filings |
| Cache | Redis | 7.0+ | Embeddings, responses |
| Task Queue | Airflow | 2.7+ | Batch processing |

#### **AI/ML**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Text Embeddings | Jina AI v3 | 1024-dim semantic vectors |
| Visual Embeddings | Nomic Embed Vision v1.5 | 768-dim visual vectors |
| LLM (Tier 1) | Google Gemini Flash | Primary generation |
| LLM (Tier 2) | Google Gemini Pro | Fallback |
| LLM (Tier 3) | Anthropic Claude 3.5 Haiku | Fallback |
| LLM (Tier 4) | OpenAI GPT-4o-mini | Last resort |

#### **Document Processing**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| PDF Parsing | PyMuPDF | Extract text, images |
| Image Processing | Pillow (PIL) | Resize, normalize |
| PDF to Image | pdf2image | Page-level conversion |

#### **Frontend**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| UI Framework | Streamlit | Interactive demo |
| File Upload | Streamlit file_uploader | Image upload |
| Deployment | Streamlit Cloud | Free hosting |

#### **DevOps**
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Containerization | Docker | Application packaging |
| Deployment | Railway | Production hosting |
| CI/CD | GitHub Actions | Automated deployment |
| Monitoring | Railway Metrics | Performance tracking |

### **Dependencies**

**New additions for multimodal:**
```toml
# pyproject.toml additions
[tool.poetry.dependencies]
nomic = "^2.0.0"           # Nomic Embed Vision SDK
PyMuPDF = "^1.23.0"        # PDF processing
pillow = "^10.0.0"         # Image handling
pdf2image = "^1.16.3"      # PDF to image conversion
apache-airflow = "^2.7.0"  # Workflow orchestration
redis = "^5.0.0"           # Caching
```

**Total new dependencies:** ~6 packages
**Size increase:** ~50MB (minimal impact)

### **API Specifications**

#### **Updated `/ask` Endpoint**

```python
POST /api/v1/ask

Request:
{
  "query": str,              # Text query (required)
  "document_type": str,      # "arxiv" or "financial" (required)
  "top_k": int,              # Number of results (default: 5)
  "ticker": str,             # Financial: ticker filter (optional)
  "filing_types": [str],     # Financial: filing types (optional)
  "image": File              # 🆕 NEW: Image for visual search (optional)
}

Response:
{
  "answer": str,
  "sources": [str],
  "chunks_used": int,
  "search_mode": str,        # "text", "visual", or "multimodal"
  "retrieval_breakdown": {   # 🆕 NEW
    "text_results": int,
    "visual_results": int,
    "fusion_method": str     # "RRF" or "text_only"
  },
  "model_used": str,
  "tokens_used": {...},
  "confidence": str
}
```

**New query parameter: `image`**
- Type: Binary file upload (multipart/form-data)
- Formats: PNG, JPG, JPEG
- Max size: 10MB
- Purpose: Visual similarity search

### **OpenSearch Index Schemas**

#### **Visual Index Schema**

```json
{
  "arxiv_visual": {
    "settings": {
      "number_of_shards": 1,
      "number_of_replicas": 0,
      "index.knn": true,
      "index.knn.algo_param.ef_search": 100
    },
    "mappings": {
      "properties": {
        "paper_id": {"type": "keyword"},
        "arxiv_id": {"type": "keyword"},
        "page_number": {"type": "integer"},
        "image_path": {"type": "text"},
        "visual_embedding": {
          "type": "knn_vector",
          "dimension": 768,
          "method": {
            "name": "hnsw",
            "space_type": "l2",
            "engine": "faiss"
          }
        },
        "extracted_at": {"type": "date"},
        "metadata": {"type": "object"}
      }
    }
  }
}
```

---

## 🚢 Deployment Plan

### **Deployment Architecture**

**Platform:** Railway (same as v1)
**Repository:** `sushiva/multimodal-rag-research-finance`
**Branch Strategy:**
- `main` → Production deployment
- `dev` → Development/testing

### **Railway Services Configuration**

```yaml
# railway.json (simplified)
{
  "services": [
    {
      "name": "fastapi-backend",
      "builder": "DOCKERFILE",
      "dockerfile_path": "Dockerfile",
      "variables": {
        "LLM_PROVIDER": "gemini",
        "GEMINI_API_KEY": "$GEMINI_API_KEY",
        "ANTHROPIC_API_KEY": "$ANTHROPIC_API_KEY",
        "OPENAI_API_KEY": "$OPENAI_API_KEY",
        "NOMIC_API_KEY": "$NOMIC_API_KEY",
        "OPENSEARCH_URL": "$OPENSEARCH_URL",
        "POSTGRES_URL": "$POSTGRES_URL",
        "REDIS_URL": "$REDIS_URL"
      }
    },
    {
      "name": "opensearch",
      "image": "opensearchproject/opensearch:2.11.0",
      "volumes": [
        {
          "mountPath": "/usr/share/opensearch/data",
          "size": "10GB"
        }
      ]
    },
    {
      "name": "postgres",
      "image": "postgres:15",
      "volumes": [
        {
          "mountPath": "/var/lib/postgresql/data",
          "size": "5GB"
        }
      ]
    },
    {
      "name": "redis",
      "image": "redis:7-alpine"
    }
  ]
}
```

### **Environment Variables**

**Required for deployment:**
```bash
# LLM API Keys
GEMINI_API_KEY=...
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...

# NEW: Visual embeddings
NOMIC_API_KEY=...

# Database URLs (Railway provides these automatically)
OPENSEARCH_URL=...
POSTGRES_URL=...
REDIS_URL=...

# Application settings
LLM_PROVIDER=gemini
ENVIRONMENT=production
LOG_LEVEL=info
```

### **Deployment Checklist**

**Pre-Deployment:**
- [ ] All tests passing locally
- [ ] Visual embeddings generated for test dataset
- [ ] OpenSearch indices created
- [ ] Environment variables configured
- [ ] Dockerfile optimized
- [ ] Dependencies locked (poetry.lock)

**Deployment Steps:**
1. Push to `main` branch
2. Railway auto-builds Docker image
3. Railway deploys to production
4. Health check verification
5. Test multimodal query end-to-end
6. Monitor logs for errors

**Post-Deployment:**
- [ ] Health endpoint responding
- [ ] Text search working
- [ ] Visual search working
- [ ] Multimodal search working
- [ ] LLM fallback working
- [ ] Metrics collected

### **Monitoring & Alerts**

**Railway Metrics:**
- CPU usage
- Memory usage
- Request latency
- Error rates

**Custom Health Checks:**
```python
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "services": {
            "opensearch": check_opensearch(),
            "postgres": check_postgres(),
            "redis": check_redis(),
            "visual_embedding_service": check_nomic()
        },
        "indices": {
            "arxiv": get_doc_count("arxiv"),
            "financial": get_doc_count("financial"),
            "arxiv_visual": get_doc_count("arxiv_visual"),
            "financial_visual": get_doc_count("financial_visual")
        }
    }
```

---

## 🎨 Portfolio Positioning

### **How to Position This Project**

#### **On GitHub README:**

```markdown
# Multimodal RAG for Research & Financial Documents

🚀 **Cutting-edge multimodal retrieval system** enabling visual similarity search across research papers and financial filings.

## 🌟 What Makes This Unique

- **Text + Visual Search**: Query by diagrams, tables, charts, and text simultaneously
- **Dual Domain**: Research papers (arXiv) + Financial documents (SEC filings)
- **Production-Ready**: Deployed on Railway with 4-tier LLM fallback (99.9% uptime)
- **Cost-Optimized**: ~$22/month with visual embeddings and caching

## 🔬 Example Queries

**Research Papers:**
- "Find papers with transformer architecture diagrams similar to this"
- "Show me papers with convergence patterns like this loss curve"

**Financial Documents:**
- "Find companies with revenue breakdowns structured like Apple's"
- "Show me balance sheets with similar asset allocation patterns"

## 🏗️ Architecture

Built from scratch (no LangChain) with:
- Nomic Embed Vision (768-dim visual embeddings)
- Jina AI (1024-dim text embeddings)
- OpenSearch (4 indices: text + visual for both domains)
- FastAPI + Railway deployment
```

#### **On Portfolio Website:**

**Project Card:**
```
🎨 Multimodal RAG Research System

Evolution of my text-based RAG system to support visual similarity search.
Users can now find research papers by diagram patterns and financial documents
by table/chart structures.

Tech: Python, Nomic Vision, OpenSearch, FastAPI, Railway
Impact: Enables new search paradigms for researchers and financial analysts
```

#### **On Resume:**

**Project Bullet:**
```
• Designed and deployed multimodal RAG system supporting text + visual search
  across 200+ documents (research papers + financial filings) using Nomic
  Embed Vision (768-dim) and custom fusion ranking, achieving sub-500ms query
  latency while maintaining production reliability (99.9% uptime, 4-tier LLM
  fallback)
```

### **Interview Talking Points**

**"Tell me about this multimodal RAG project"**

> "I started with a production text-based RAG system [v1], which worked well but had limitations - researchers couldn't search by diagrams, financial analysts couldn't find similar table structures.
>
> So I researched multimodal embeddings, experimented with ColPali and Nomic Vision, and built a second system from scratch. The key challenges were:
>
> 1. **Dual visual patterns**: Research papers have diagrams/equations vs financial docs have tables/charts - required different extraction strategies
>
> 2. **Fusion ranking**: How do you combine text similarity scores (from BM25/vector) with visual similarity scores? I extended Reciprocal Rank Fusion to handle multimodal results.
>
> 3. **Cost optimization**: Visual embeddings add storage/compute costs. I used Redis caching and batch processing via Airflow to keep total costs under $25/month.
>
> The result: Users can now query 'find papers with transformer diagrams like this' or 'show me companies with similar revenue structures to Apple' - capabilities that didn't exist before.
>
> It's deployed on Railway, and I kept the original text system running to show I can maintain stable production while innovating."

**"Why didn't you use LangChain?"**

> "LangChain is great for prototyping, but I built from scratch because:
>
> 1. **Consistency**: My v1 system uses custom service layers, direct OpenSearch access. Adding LangChain would create architectural inconsistency.
>
> 2. **Performance**: For production, I wanted full control over the retrieval pipeline. LangChain's abstractions add overhead I didn't need.
>
> 3. **Learning**: Building from scratch deepened my understanding of multimodal retrieval - how to normalize visual vs text similarities, optimal fusion strategies, etc.
>
> 4. **Portfolio value**: Shows I can implement advanced AI concepts from first principles, not just chain together libraries.
>
> That said, I referenced LangChain's multimodal patterns for inspiration. It's a great learning resource."

**"How does visual similarity search work?"**

> "The pipeline is:
>
> 1. **Extraction**: PyMuPDF extracts each PDF page as a 150 DPI image
>
> 2. **Embedding**: Nomic Embed Vision v1.5 generates 768-dimensional vectors representing visual patterns - shapes, layouts, structures
>
> 3. **Indexing**: Store in OpenSearch with HNSW algorithm for fast approximate nearest neighbor search
>
> 4. **Query**: User uploads an image (or we use a text-to-image model), generate its embedding, search OpenSearch for similar vectors
>
> 5. **Fusion**: If user provides both text and image, we run two searches and merge results using Reciprocal Rank Fusion - gives each result a score based on its rank in both lists.
>
> For example, a paper ranked #1 in visual search and #5 in text search gets a higher combined score than one ranked #3 in both.
>
> The key insight: Visual similarity complements text similarity. A paper might not mention 'transformer' explicitly but have a clear transformer diagram."

---

## 🛡️ Risk Mitigation

### **Identified Risks & Mitigation Strategies**

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Visual embeddings too expensive** | High | Low | Use free tier (1M tokens/month), cache embeddings in Redis |
| **Storage costs exceed budget** | Medium | Medium | Start with 20-30 papers, scale gradually |
| **Visual search quality poor** | High | Medium | Test with multiple models (Nomic, CLIP), use best performer |
| **Deployment complexity** | Medium | Low | Use Railway (same as v1), incremental deployment |
| **Breaking v1 during development** | High | Low | Separate repos, no code sharing initially |
| **Performance degradation** | Medium | Medium | Benchmark each phase, optimize before deploying |
| **PDF extraction fails for some papers** | Low | Medium | Robust error handling, manual fallback |
| **Image upload security issues** | Medium | Low | Validate file types, size limits, sanitize uploads |

### **Contingency Plans**

**If visual embeddings are too slow:**
- Pre-generate and cache all embeddings
- Use smaller image sizes (reduce DPI from 150 to 100)
- Batch process overnight via Airflow

**If costs exceed $30/month:**
- Reduce OpenSearch replicas to 0 (demo doesn't need HA)
- Use Railway free tier for dev/staging
- Deploy only when actively job hunting

**If visual search quality is poor:**
- Try CLIP instead of Nomic
- Experiment with different image preprocessing (grayscale, edge detection)
- Focus on specific visual patterns (diagrams only, not full pages)

**If deployment fails:**
- Fall back to v1 (production stays up)
- Debug locally with docker-compose
- Use Railway's rollback feature

---

## 📊 Success Metrics

### **Technical Metrics**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **Query Latency** | <500ms | OpenSearch response times |
| **Visual Search Precision@5** | >70% | Manual evaluation on test set |
| **Multimodal Fusion Improvement** | +20% vs text-only | A/B testing |
| **Embedding Generation Time** | <5s per page | Batch processing logs |
| **Index Size** | <2GB for 100 papers | OpenSearch metrics |
| **Uptime** | >99.5% | Railway monitoring |
| **Cost per Query** | <$0.002 | Monthly cost / query count |

### **Portfolio Metrics**

| Metric | Target | How to Measure |
|--------|--------|----------------|
| **GitHub Stars** | 10+ | Repo stars |
| **Live Demo Users** | 50+ | Streamlit analytics |
| **LinkedIn Engagement** | 100+ views | Post analytics |
| **Interview Conversions** | 5+ interviews | Self-reported |
| **Recruiter Interest** | 3+ inbound messages | LinkedIn |

### **Learning Metrics**

| Skill | Before | After | Evidence |
|-------|--------|-------|----------|
| **Multimodal AI** | Novice | Intermediate | Implemented production system |
| **Vector Search** | Intermediate | Advanced | Custom fusion ranking |
| **Airflow** | Novice | Intermediate | Built batch processing DAG |
| **Redis Caching** | Novice | Intermediate | Embedding cache implementation |
| **Cost Optimization** | Intermediate | Advanced | <$25/month with 4 indices |

---

## 📚 Appendix

### **References & Learning Resources**

**Multimodal RAG:**
- [ColPali Paper](https://arxiv.org/abs/2407.01449) - Document understanding with Vision Language Models
- [Nomic Embed Vision](https://blog.nomic.ai/posts/nomic-embed-vision) - Open vision embeddings
- [LangChain Multimodal](https://python.langchain.com/docs/use_cases/multimodal) - Reference patterns

**Visual Embeddings:**
- [CLIP Paper](https://arxiv.org/abs/2103.00020) - Learning transferable visual models
- [Nomic Atlas](https://atlas.nomic.ai/) - Visual exploration of embeddings

**Vector Search:**
- [OpenSearch k-NN](https://opensearch.org/docs/latest/search-plugins/knn/) - Configuration guide
- [HNSW Algorithm](https://arxiv.org/abs/1603.09320) - Efficient similarity search

**Airflow:**
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Batch Processing Patterns](https://www.astronomer.io/guides/dag-best-practices)

### **Sample Queries for Testing**

**arXiv Papers:**
```
1. Text: "Papers discussing transformer architectures"
   Expected: Attention is All You Need, BERT, GPT papers

2. Visual: [Upload transformer diagram image]
   Expected: Papers with similar encoder-decoder architectures

3. Multimodal: "Papers about GANs" + [Upload GAN diagram]
   Expected: High-ranked papers with both GAN discussion AND visual architecture
```

**Financial Documents:**
```
1. Text: "Companies discussing AI revenue growth"
   Expected: NVDA, MSFT 10-Ks

2. Visual: [Upload revenue breakdown table]
   Expected: Companies with similar business segment structures

3. Multimodal: "Tech companies Q4 results" + [Upload revenue chart]
   Expected: Companies with similar quarterly patterns
```

### **File Structure (Complete)**

```
multimodal-rag-research-finance/
├── src/
│   ├── services/
│   │   ├── vision/                    # 🆕 NEW
│   │   │   ├── __init__.py
│   │   │   ├── pdf_extractor.py
│   │   │   ├── nomic_client.py
│   │   │   └── image_processor.py
│   │   ├── search/
│   │   │   ├── opensearch.py          # ✅ From v1
│   │   │   └── multimodal_retriever.py # 🆕 NEW
│   │   ├── gemini/                    # ✅ From v1
│   │   ├── anthropic/                 # ✅ From v1
│   │   └── openai/                    # ✅ From v1
│   ├── routers/
│   │   └── ask.py                     # 📝 Updated
│   ├── ingestion/
│   │   ├── arxiv_ingestion.py         # ✅ From v1
│   │   ├── financial_ingestion.py     # ✅ From v1
│   │   └── visual_embeddings.py       # 🆕 NEW
│   └── config.py                      # 📝 Updated
│
├── airflow/
│   └── dags/
│       └── visual_embedding_pipeline.py # 🆕 NEW
│
├── streamlit_app.py                   # 📝 Updated (image upload)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                     # 📝 Updated (new deps)
├── poetry.lock
├── railway.json
├── .env.example
├── README.md                          # 📝 Updated (multimodal focus)
├── MULTIMODAL_RAG_PLAN.md            # This document
└── docs/
    ├── API.md
    ├── DEPLOYMENT.md
    └── VISUAL_SEARCH.md               # 🆕 NEW
```

---

## ✅ Next Steps

### **Immediate Actions (Before Development)**

1. **Review & Approve Plan**
   - [ ] Review this comprehensive plan
   - [ ] Approve architecture decisions
   - [ ] Confirm budget ($22-25/month acceptable)
   - [ ] Set start date

2. **Environment Setup**
   - [ ] Get Nomic API key (free tier)
   - [ ] Install local dependencies
   - [ ] Set up dev environment

3. **Test Dataset Preparation**
   - [ ] Select 20-30 arXiv papers for Phase 1
   - [ ] Select 10-20 financial docs for Phase 5
   - [ ] Download and organize PDFs

4. **Repository Setup**
   - [ ] Create new GitHub repo
   - [ ] Fork code from v1
   - [ ] Update README

### **Development Sequence**

**Week 1:** Visual extraction & embeddings (Phase 1)
**Week 2:** OpenSearch visual indices (Phase 2)
**Week 3:** Multimodal retrieval (Phase 3)
**Week 4:** Airflow + Financial docs (Phases 4-5)
**Week 5:** UI updates & deployment (Phase 6)

### **Questions to Answer Before Starting**

1. **Start date?** (Recommend: After Airflow + Redis are added to v1)
2. **Test dataset?** (20 arXiv + 10 Financial confirmed?)
3. **Budget approval?** ($22-25/month for v2 acceptable?)
4. **Repository name final?** (`multimodal-rag-research-finance` confirmed?)

---

## 📞 Contact & Support

**Project Owner:** Sudhir Shivaram
**GitHub:** [@sushiva](https://github.com/sushiva)
**Email:** Shivaram.Sudhir@gmail.com
**LinkedIn:** [linkedin.com/in/sudhirshivaram](https://linkedin.com/in/sudhirshivaram)

---

**Document Version:** 1.0
**Last Updated:** 2024-12-12
**Status:** Planning Complete - Awaiting Approval

---

*This is a comprehensive implementation plan for building a production-grade multimodal RAG system. All cost estimates, timelines, and technical specifications are based on current research and v1 system performance data.*
