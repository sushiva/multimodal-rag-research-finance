# System Architecture

## Overview

This document explains the microservices architecture, design decisions, and rationale behind the multimodal RAG system.

## Design Philosophy

### Core Principles

1. **Separation of Concerns**: Each microservice has a single, well-defined responsibility
2. **Scalability**: Services can scale independently based on load
3. **Resilience**: Failure in one service doesn't cascade to others
4. **Maintainability**: Clean interfaces, clear boundaries, comprehensive logging
5. **Security**: Defense in depth, authentication, validation at every layer

### Why Microservices?

**Rationale:**
- **Independent Scaling**: Visual embedding generation is CPU-intensive; text search is I/O-intensive. Different resource profiles.
- **Technology Flexibility**: Can upgrade OpenSearch without touching FastAPI code
- **Team Scalability**: Different developers can work on different services
- **Fault Isolation**: If Redis fails, system degrades gracefully (no cache) but stays operational
- **Deployment Flexibility**: Can deploy services on different infrastructure (GPU for embeddings, fast SSD for database)

**Trade-offs:**
- Complexity: More moving parts than monolith
- Network latency: Inter-service communication overhead
- Data consistency: Eventual consistency challenges

**Why we accept the trade-offs:**
- This system is read-heavy (95% reads, 5% writes), so consistency is manageable
- Network latency is minimal when services are co-located (Docker/Railway)
- Complexity is offset by better scalability and resilience

## Microservices Architecture

### Service Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                         Client Layer                          │
│  (Streamlit UI, API clients, Mobile apps)                     │
└────────────────────────┬─────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                   API Gateway / Load Balancer                 │
│  (Railway, Nginx - handles SSL, rate limiting)                │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                     FastAPI Service                           │
│  Responsibilities:                                            │
│  - Request routing                                            │
│  - Input validation (Pydantic)                                │
│  - Authentication & authorization                             │
│  - LLM orchestration (4-tier fallback)                        │
│  - Response formatting                                        │
│                                                               │
│  Components:                                                  │
│  ├─ /routers          (API endpoints)                        │
│  ├─ /services/llm     (Gemini, Claude, OpenAI clients)       │
│  ├─ /services/search  (Multimodal retriever)                 │
│  ├─ /services/vision  (Visual embedding client)              │
│  ├─ /services/cache   (Redis client)                         │
│  ├─ /models           (Pydantic schemas)                     │
│  ├─ /core             (Config, logging, security)            │
│  └─ /utils            (Shared utilities)                     │
└──────┬────────────┬────────────┬────────────┬────────────────┘
       │            │            │            │
       │            │            │            │
   ┌───▼───┐    ┌───▼────┐   ┌──▼─────┐  ┌──▼────────┐
   │OpenSearch│ │PostgreSQL│ │ Redis   │  │  Airflow  │
   │ Service  │ │ Service  │ │ Service │  │  Service  │
   └──────────┘ └──────────┘ └─────────┘  └───────────┘
```

### Service Responsibilities

#### 1. FastAPI Service (Application Layer)

**What:**
HTTP API server providing RESTful endpoints for multimodal RAG queries.

**Why:**
- Single entry point for all client interactions
- Enforces business logic and security
- Coordinates multiple backing services
- Provides clear, documented API (OpenAPI/Swagger)

**How:**
- Async Python for high concurrency
- Pydantic for request/response validation
- Dependency injection for service composition
- Structured logging for observability

**Dependencies:**
- OpenSearch (vector search)
- PostgreSQL (metadata)
- Redis (caching)
- External APIs (Gemini, Claude, OpenAI, Nomic)

**Failure Modes:**
- If OpenSearch fails: Return 503, log error
- If PostgreSQL fails: Return 503, log error
- If Redis fails: Degrade gracefully (no cache, slower)
- If LLM fails: Try next tier (4-tier fallback)

**Performance Targets:**
- <100ms latency (excluding LLM generation)
- 100+ req/sec throughput
- 99.9% uptime

---

#### 2. OpenSearch Service (Vector Search Layer)

**What:**
Distributed search engine storing and querying text and visual embeddings.

**Why:**
- Specialized for vector similarity search (HNSW algorithm)
- Scales horizontally (sharding, replicas)
- Fast approximate nearest neighbor search
- Built-in relevance scoring

**How:**
- 4 indices:
  - `arxiv` (text embeddings, 1024-dim)
  - `financial` (text embeddings, 1024-dim)
  - `arxiv_visual` (visual embeddings, 768-dim)
  - `financial_visual` (visual embeddings, 768-dim)
- HNSW index for sub-100ms vector search
- k-NN plugin for similarity search
- No data replication (demo doesn't need HA)

**Dependencies:**
- None (self-contained)

**Data Model:**
```json
{
  "arxiv_visual": {
    "paper_id": "string",
    "arxiv_id": "string",
    "page_number": "integer",
    "visual_embedding": "knn_vector[768]",
    "image_path": "string",
    "metadata": "object"
  }
}
```

**Failure Modes:**
- If index fails: Return empty results, log error
- If query times out: Return partial results

**Performance Targets:**
- <50ms query latency (vector search)
- <2GB index size (100 papers)

---

#### 3. PostgreSQL Service (Relational Data Layer)

**What:**
Relational database storing structured metadata about papers and filings.

**Why:**
- Complex joins (papers ↔ authors ↔ categories)
- ACID transactions (data consistency)
- Full-text search (title, abstract)
- Referential integrity (foreign keys)

**How:**
- Normalized schema:
  - `papers` table (arxiv_id, title, abstract, published_date)
  - `authors` table (name, affiliation)
  - `filings` table (ticker, filing_type, filing_date, url)
  - `embeddings_metadata` table (paper_id, embedding_version, created_at)
- Indexes on frequently queried columns (arxiv_id, ticker, filing_date)

**Dependencies:**
- None (self-contained)

**Failure Modes:**
- If connection fails: Return 503, retry with backoff
- If query times out: Return error, log slow query

**Performance Targets:**
- <10ms query latency (indexed lookups)
- 1000+ rows/sec insert rate

---

#### 4. Redis Service (Caching Layer)

**What:**
In-memory key-value store for caching embeddings and query responses.

**Why:**
- Avoid re-generating expensive embeddings
- Reduce latency for repeated queries
- Rate limiting (track API usage per user)
- Session management (future: user authentication)

**How:**
- Cache strategies:
  - **Embedding cache**: Key = `embedding:{text_hash}`, Value = embedding vector, TTL = 7 days
  - **Response cache**: Key = `response:{query_hash}`, Value = full response, TTL = 1 hour
  - **Rate limit**: Key = `ratelimit:{user_id}`, Value = request count, TTL = 1 minute
- LRU eviction policy
- Persistence disabled (cache can rebuild)

**Dependencies:**
- None (self-contained)

**Failure Modes:**
- If Redis fails: System continues without cache (slower, but functional)
- If cache miss: Regenerate data, store in cache

**Performance Targets:**
- <1ms read latency
- <5ms write latency
- 90%+ cache hit rate (steady state)

---

#### 5. Airflow Service (Batch Processing Layer)

**What:**
Workflow orchestration for batch jobs (visual embedding generation, data ingestion).

**Why:**
- Scheduled jobs (daily paper ingestion, weekly reindexing)
- Complex workflows (PDF → Image → Embedding → Index)
- Retry logic, error handling
- Monitoring, alerting

**How:**
- DAGs (Directed Acyclic Graphs):
  - `visual_embedding_pipeline`: PDF → extract images → generate embeddings → index to OpenSearch
  - `arxiv_ingestion`: Fetch new papers → process → store metadata → generate text embeddings
  - `reindex_pipeline`: Rebuild indices with updated embeddings
- CeleryExecutor for parallel task execution
- PostgreSQL for DAG metadata

**Dependencies:**
- FastAPI services (calls APIs to ingest data)
- OpenSearch (indexes embeddings)
- PostgreSQL (stores metadata)

**Failure Modes:**
- If task fails: Retry 3x with exponential backoff
- If DAG fails: Send alert, log error, mark as failed

**Performance Targets:**
- Process 100 papers in <30 minutes
- <1% task failure rate

---

## Data Flow: End-to-End Query

### Scenario: User queries "Find papers with transformer diagrams"

**Step-by-Step:**

1. **Client → FastAPI**
   ```
   POST /api/v1/ask
   Body: {
     "query": "Find papers with transformer diagrams",
     "image": <uploaded_diagram.png>,
     "document_type": "arxiv"
   }
   ```

2. **FastAPI → Redis (Cache Check)**
   ```python
   cache_key = f"response:{hash(query + image_hash)}"
   cached_response = redis.get(cache_key)
   if cached_response:
       return cached_response  # Cache hit, return immediately
   ```

3. **FastAPI → Vision Service (Generate Visual Embedding)**
   ```python
   # If image provided
   visual_embedding = await nomic_client.embed_images([image])
   # Cache the embedding
   redis.set(f"embedding:{image_hash}", visual_embedding, ttl=7*24*3600)
   ```

4. **FastAPI → Search Service (Multimodal Retrieval)**
   ```python
   # Text search
   text_results = await opensearch.hybrid_search(
       index="arxiv",
       query="Find papers with transformer diagrams",
       top_k=10
   )

   # Visual search
   visual_results = await opensearch.vector_search(
       index="arxiv_visual",
       embedding=visual_embedding,
       top_k=10
   )

   # Fusion ranking (RRF)
   combined_results = reciprocal_rank_fusion(text_results, visual_results)
   ```

5. **FastAPI → PostgreSQL (Enrich Metadata)**
   ```python
   paper_ids = [r['paper_id'] for r in combined_results]
   metadata = await postgres.query(
       "SELECT * FROM papers WHERE paper_id IN (%s)",
       paper_ids
   )
   ```

6. **FastAPI → LLM Service (Generate Answer)**
   ```python
   # 4-tier fallback
   try:
       answer = await gemini_client.generate(prompt)
   except:
       try:
           answer = await claude_client.generate(prompt)
       except:
           answer = await openai_client.generate(prompt)
   ```

7. **FastAPI → Redis (Cache Response)**
   ```python
   redis.set(cache_key, response, ttl=3600)  # 1 hour TTL
   ```

8. **FastAPI → Client (Return Response)**
   ```json
   {
     "answer": "...",
     "sources": [...],
     "search_mode": "multimodal",
     "retrieval_breakdown": {
       "text_results": 10,
       "visual_results": 10,
       "fusion_method": "RRF"
     },
     "latency_ms": 450
   }
   ```

**Total Latency Breakdown:**
- Cache check: 1ms
- Visual embedding: 100ms (or 0ms if cached)
- Text search: 30ms
- Visual search: 40ms
- Fusion ranking: 10ms
- Metadata enrichment: 5ms
- LLM generation: 2000ms (varies)
- **Total: ~2200ms** (90% is LLM generation)

---

## Security Architecture

### Threat Model

**Assets to Protect:**
- API keys (Gemini, Claude, OpenAI, Nomic)
- User data (queries, uploads)
- System availability (prevent DoS)

**Threats:**
- Unauthorized access
- Data exfiltration
- Denial of service
- Injection attacks (SQL, prompt)

### Security Layers

#### Layer 1: Network Security
- **HTTPS Only**: All traffic encrypted (Railway enforces)
- **CORS Configuration**: Only allow specific origins
- **Firewall**: Only expose necessary ports (80, 443)

#### Layer 2: Authentication & Authorization
- **API Keys**: JWT tokens for client authentication
- **Rate Limiting**: 60 requests/minute per user (Redis-based)
- **IP Whitelisting**: For admin endpoints

#### Layer 3: Input Validation
- **Pydantic Models**: Type-safe request validation
- **File Upload Security**:
  - Max size: 10MB
  - Allowed types: PNG, JPG, JPEG
  - Virus scanning (future: ClamAV integration)
  - Filename sanitization

#### Layer 4: Data Security
- **SQL Injection Prevention**: Parameterized queries only
- **Prompt Injection Prevention**: Sanitize user input before LLM
- **Secrets Management**: Environment variables, never in code

#### Layer 5: Monitoring & Logging
- **Audit Logging**: Log all API requests (who, what, when)
- **Error Tracking**: Sentry for error monitoring
- **Anomaly Detection**: Alert on unusual patterns

**Implementation Priority:**
1. ✅ Phase 1: Input validation, HTTPS
2. Phase 2: API authentication, rate limiting
3. Phase 3: Audit logging, monitoring
4. Phase 4: Advanced (virus scanning, anomaly detection)

---

## Observability

### Logging Strategy

**Log Levels:**
- `DEBUG`: Detailed info for development
- `INFO`: Normal operations (request received, cache hit)
- `WARNING`: Unusual but handled (LLM tier failover)
- `ERROR`: Errors requiring attention (database timeout)
- `CRITICAL`: System failure (all services down)

**Structured Logging:**
```python
logger.info(
    "Multimodal query processed",
    extra={
        "user_id": user_id,
        "query": query,
        "document_type": doc_type,
        "search_mode": "multimodal",
        "latency_ms": 450,
        "cache_hit": False
    }
)
```

**Log Aggregation:**
- Railway Metrics (production)
- Local: Console + file (development)

### Monitoring

**Key Metrics:**
- **Request Metrics**: Rate, latency, errors
- **Service Health**: Uptime, response time
- **Cache Metrics**: Hit rate, eviction rate
- **LLM Metrics**: Tier usage, token consumption
- **Database Metrics**: Query time, connection pool

**Dashboards:**
- Railway (CPU, memory, latency)
- Custom (business metrics: cache hit rate, LLM tier distribution)

**Alerts:**
- Error rate >1%
- Latency >1 second (p95)
- OpenSearch health degraded
- Redis eviction rate >50%

---

## Deployment Architecture

### Local Development

```
Docker Compose:
- FastAPI (localhost:8000)
- OpenSearch (localhost:9200)
- PostgreSQL (localhost:5432)
- Redis (localhost:6379)
- Airflow (localhost:8080)

Advantages:
- Full stack on laptop
- Fast iteration
- No deployment cost

Disadvantages:
- Resource intensive (8GB+ RAM)
- Not production-like
```

### Production (Railway)

```
Railway Services:
- FastAPI (auto-scaled, $8/month)
- OpenSearch (2GB RAM, $8-10/month)
- PostgreSQL (1GB storage, $3/month)
- Redis (512MB RAM, $3-4/month)
- Airflow (future: separate instance)

Advantages:
- Production-grade (HA, backups)
- Auto-scaling
- Managed (no ops burden)

Disadvantages:
- Cost ($22-25/month)
- Less control than K8s
```

### Why Railway > Kubernetes?

**For this project:**
- ✅ Railway is simpler (no K8s complexity)
- ✅ Railway is cheaper ($25/month vs $50+ for managed K8s)
- ✅ Railway auto-scales (no manual HPA configuration)
- ✅ Railway has CI/CD built-in

**When K8s makes sense:**
- 20+ microservices
- Multi-region deployment
- Advanced networking (service mesh, istio)
- Enterprise requirements (compliance, audit)

**For portfolio demo:**
Railway demonstrates production deployment skills without over-engineering.

---

## Future Enhancements

### Scalability
- **Horizontal Scaling**: Add more FastAPI instances behind load balancer
- **Read Replicas**: PostgreSQL read replicas for query load
- **OpenSearch Sharding**: Shard large indices across multiple nodes

### Reliability
- **Circuit Breakers**: Prevent cascading failures (Resilience4j)
- **Bulkheads**: Isolate thread pools per service
- **Chaos Engineering**: Test failure scenarios (Chaos Monkey)

### Features
- **Multi-Language Support**: Embed multilingual papers
- **Real-Time Updates**: WebSocket for streaming results
- **Analytics Dashboard**: Query patterns, popular papers
- **A/B Testing**: Compare fusion algorithms

---

## References

**Microservices:**
- "Building Microservices" by Sam Newman
- "Designing Data-Intensive Applications" by Martin Kleppmann

**Vector Search:**
- HNSW Algorithm: https://arxiv.org/abs/1603.09320
- OpenSearch k-NN: https://opensearch.org/docs/latest/search-plugins/knn/

**Security:**
- OWASP Top 10: https://owasp.org/www-project-top-ten/
- API Security Best Practices: https://owasp.org/www-project-api-security/

**Deployment:**
- 12-Factor App: https://12factor.net/
- Railway Docs: https://docs.railway.app/

---

**Document Version**: 1.0
**Last Updated**: December 2024
**Status**: Living Document (will evolve as system grows)
