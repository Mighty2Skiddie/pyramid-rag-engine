# Vexoo Labs AI Assignment — Industry-Grade Web Platform

**Complete Product & Architecture Planning Document**

| Field | Detail |
|-------|--------|
| Document Type | Product Design + System Architecture |
| Scope | Full-stack web platform to showcase, deploy, and demonstrate the AI assignment as a live product |
| Audience | Hiring managers, engineers, technical evaluators |

---

## 1. Product Vision & Goals

### What We Are Building

A professional, interactive web platform that presents the Vexoo Labs AI assignment not as a script submission, but as a **live, working AI product**. Visitors can upload documents, query the Knowledge Pyramid in real time, and see the GSM8K-trained reasoning model solve math problems — all through a polished browser interface.

### Why This Matters

Submitting a zip file of Python scripts is what everyone does. Deploying it as a working product immediately signals:

- You think beyond code — you think in **systems**
- You can take an AI model from notebook to **production**
- You understand UX, APIs, and deployment — not just ML

### Core Goals

- Make both AI features (RAG pipeline + reasoning model) accessible via a clean web UI
- Present the system architecture visually so evaluators understand your thinking
- Make the platform fast, responsive, and impressive on first visit
- Keep infrastructure cost near zero (student/assignment budget)

### Target Audience

- Vexoo Labs hiring team and technical evaluators
- Anyone reviewing your portfolio
- Potential future employers browsing your GitHub or personal site

---

## 2. Product Structure — What Pages & Features Exist

The platform has five distinct sections:

```
┌─────────────────────────────────────────────┐
│            VEXOO AI PLATFORM                │
│                                             │
│  1. Landing Page      (Hero + Overview)     │
│  2. Document Lab      (Part 1 — RAG Demo)   │
│  3. Reasoning Lab     (Part 2 — GSM8K Demo) │
│  4. Architecture Page (System Design)       │
│  5. About / Docs      (README + Report)     │
└─────────────────────────────────────────────┘
```

---

## 3. System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        BROWSER (Client)                              │
│                                                                      │
│   Next.js 14 App Router — React Server + Client Components          │
│   Tailwind CSS + shadcn/ui — Design System                          │
│   Framer Motion — Animations                                         │
└────────────────────────────┬─────────────────────────────────────────┘
                             │  HTTPS REST / JSON
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        API LAYER                                     │
│                                                                      │
│   FastAPI (Python) — Hosted on Render / Railway                     │
│   Two core routers:                                                  │
│     /api/pyramid  →  Document ingestion + query                     │
│     /api/reason   →  Math reasoning model inference                 │
└────────────┬─────────────────────────────────┬───────────────────────┘
             │                                 │
             ▼                                 ▼
┌─────────────────────────┐       ┌──────────────────────────┐
│   PYRAMID ENGINE        │       │   REASONING ENGINE       │
│                         │       │                          │
│   Part 1 Python         │       │   Fine-tuned LLaMA /     │
│   modules running       │       │   GPT-2 adapter running  │
│   as service            │       │   as inference service   │
└─────────────────────────┘       └──────────────────────────┘
             │                                 │
             ▼                                 ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                    │
│                                                                      │
│   In-memory session store (Redis on free tier)                      │
│   Static assets on Cloudflare CDN                                   │
│   Model weights on HuggingFace Hub (pulled at startup)              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Page-by-Page Design Plan

### Page 1 — Landing Page

**Purpose**: Make an instant strong impression. Communicate what the project is in under 10 seconds.

```
┌─────────────────────────────────────────────┐
│  NAVBAR: Logo | Features | Docs | GitHub    │
├─────────────────────────────────────────────┤
│                                             │
│  HERO SECTION                               │
│  ─────────────                              │
│  Large headline: "AI Document Intelligence  │
│  + Math Reasoning — Live Demo"              │
│                                             │
│  Subtext: 2-line explanation of the project │
│                                             │
│  [Try Document Lab]  [Try Reasoning Lab]    │
│  (two CTA buttons, side by side)            │
│                                             │
│  Animated background: subtle particle       │
│  graph or flowing gradient                  │
│                                             │
├─────────────────────────────────────────────┤
│  FEATURE CARDS ROW (3 cards)                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐    │
│  │Knowledge │ │ GSM8K    │ │Reasoning │    │
│  │ Pyramid  │ │Fine-Tune │ │ Adapter  │    │
│  │  RAG     │ │  Model   │ │  Design  │    │
│  └──────────┘ └──────────┘ └──────────┘    │
├─────────────────────────────────────────────┤
│  HOW IT WORKS (3-step visual flow)          │
│  Upload Doc → Build Pyramid → Query & Get   │
├─────────────────────────────────────────────┤
│  TECH STACK BADGES                          │
│  Python | FastAPI | LLaMA | LoRA | Next.js  │
├─────────────────────────────────────────────┤
│  FOOTER: GitHub | LinkedIn | Report PDF     │
└─────────────────────────────────────────────┘
```

**Key design decisions:**

- Dark theme (deep navy or near-black) with electric blue/cyan accent — signals AI/tech product
- Headline uses a gradient text effect
- CTA buttons use contrasting accent color with hover animation
- No clutter — every element has a purpose

---

### Page 2 — Document Lab (Part 1 Demo)

**Purpose**: Let users upload a document, see the Knowledge Pyramid get built, and query it live.

```
┌──────────────────────────────────────────────────────┐
│  PAGE HEADER: "Document Intelligence Lab"            │
│  Subtitle: "Upload → Ingest → Query the Pyramid"    │
├──────────────────────────────────────────────────────┤
│                                                      │
│  STEP 1: UPLOAD                                      │
│  ┌────────────────────────────────────────┐          │
│  │  Drag & drop zone  OR  paste text box  │          │
│  │  "Drop a .txt or .pdf file here"       │          │
│  │  Max size: 500KB                       │          │
│  │  [Or use sample document]  ← fallback  │          │
│  └────────────────────────────────────────┘          │
│                                                      │
│  STEP 2: PYRAMID VISUALIZATION                       │
│  (appears after ingestion, animated build-up)        │
│  ┌────────────────────────────────────────┐          │
│  │         ▲  L4: Distilled Knowledge     │          │
│  │        ▲▲▲  L3: Category Labels        │          │
│  │      ▲▲▲▲▲  L2: Chunk Summaries        │          │
│  │    ▲▲▲▲▲▲▲  L1: Raw Text Chunks        │          │
│  │                                        │          │
│  │  Click any layer → see its content     │          │
│  └────────────────────────────────────────┘          │
│                                                      │
│  STEP 3: QUERY                                       │
│  ┌────────────────────────────────────────┐          │
│  │  [Search box: "Ask anything about..."] │          │
│  │  [Submit Query]                        │          │
│  └────────────────────────────────────────┘          │
│                                                      │
│  STEP 4: RESULTS PANEL                               │
│  ┌────────────────────────────────────────┐          │
│  │  Best Match Found at: Layer 2          │          │
│  │  Confidence Score: 87%                 │          │
│  │  ─────────────────────────────         │          │
│  │  Matched Text: "..."                   │          │
│  │  Category: Technical                   │          │
│  │  Keywords: [ai, model, training, ...]  │          │
│  └────────────────────────────────────────┘          │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Interactive elements:**

- Pyramid diagram is clickable — clicking a layer shows all chunks at that level in a side panel
- Results panel shows which layer the answer came from, with a visual highlight on that pyramid level
- "Use Sample Document" button pre-loads a curated text so evaluators don't need to upload anything
- Processing state shows an animated loading indicator with step labels: *"Chunking... Building L1... Building L2..."*

---

### Page 3 — Reasoning Lab (Part 2 Demo)

**Purpose**: Let users type a math problem and see the fine-tuned model solve it step by step.

```
┌──────────────────────────────────────────────────────┐
│  PAGE HEADER: "Math Reasoning Lab"                   │
│  Subtitle: "LLaMA 3.2 1B fine-tuned on GSM8K"       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  MODEL INFO CARD                                     │
│  ┌──────────────────────────────────────┐            │
│  │  Base Model: LLaMA 3.2 1B            │            │
│  │  Fine-tuning: LoRA (r=8, alpha=16)   │            │
│  │  Dataset: GSM8K (3000 samples)       │            │
│  │  Eval Accuracy: XX%  ← real number   │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  INPUT PANEL                                         │
│  ┌──────────────────────────────────────┐            │
│  │  Textarea: "Type a math word problem"│            │
│  │                                      │            │
│  │  [Sample Problems]: Easy | Med | Hard│            │
│  │  [Solve →]                           │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  OUTPUT PANEL (streaming, token by token)            │
│  ┌──────────────────────────────────────┐            │
│  │  Model Reasoning:                    │            │
│  │  "Step 1: John has 5 apples..."      │            │
│  │  "Step 2: He gives away 2..."        │            │
│  │  "Step 3: 5 - 2 = 3"                │            │
│  │  ────────────────────                │            │
│  │  #### Answer: 3                      │            │
│  └──────────────────────────────────────┘            │
│                                                      │
│  METRICS SIDEBAR                                     │
│  Inference time: 1.2s                                │
│  Tokens generated: 87                                │
│  Model confidence: shown as progress bar             │
│                                                      │
└──────────────────────────────────────────────────────┘
```

**Key UX decisions:**

- Response streams token-by-token (typewriter effect) — makes the model feel alive
- Sample problems are pre-loaded so evaluators can test instantly
- If the full LLaMA model is too heavy for free hosting, a smaller model (GPT-2 fine-tuned or TinyLLaMA) serves as a drop-in substitute with a clear label

---

### Page 4 — Architecture Page

**Purpose**: Show your system thinking. This page is what separates you from candidates who just submit code.

```
┌──────────────────────────────────────────────────────┐
│  PAGE HEADER: "System Architecture"                  │
│  Subtitle: "How this platform is designed & built"   │
├──────────────────────────────────────────────────────┤
│                                                      │
│  INTERACTIVE ARCHITECTURE DIAGRAM                    │
│  (SVG or React Flow diagram, zoomable)               │
│  Shows: Browser → API → Engines → Data layer         │
│  Hovering a component shows its tech stack           │
│                                                      │
├──────────────────────────────────────────────────────┤
│  PYRAMID DEEP DIVE                                   │
│  Visual 4-layer pyramid with annotations             │
│  Each layer expandable to show its logic             │
├──────────────────────────────────────────────────────┤
│  TRAINING PIPELINE FLOWCHART                         │
│  GSM8K → Tokenizer → LoRA → Training Loop → Eval    │
│  Horizontal flow diagram with step labels            │
├──────────────────────────────────────────────────────┤
│  BONUS: REASONING ADAPTER DESIGN                     │
│  Router diagram showing query classification         │
│  and handler routing                                 │
├──────────────────────────────────────────────────────┤
│  TECH STACK TABLE                                    │
│  Frontend / Backend / AI / Infra columns             │
│  Each tech with logo, name, and one-line reason      │
└──────────────────────────────────────────────────────┘
```

---

### Page 5 — About & Docs

**Purpose**: Professional documentation and downloadable report.

**Sections:**

- Brief bio paragraph (who built this and why)
- Embedded or downloadable one-page PDF report (the assignment summary doc)
- README rendered as clean HTML (not raw markdown)
- GitHub repository link with badge (stars, last commit)
- LinkedIn and contact link
- Assignment checklist showing what was completed (visual checkmarks)

---

## 5. Frontend Architecture

### Technology: Next.js 14 with App Router

**Why Next.js over plain React:**

- Server-side rendering gives fast initial page loads — important for first impressions
- API routes allow lightweight BFF (Backend For Frontend) layer
- File-based routing keeps project structure clean
- Built-in image optimization and static asset handling

### Design System

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Framework | Next.js 14 | Routing, SSR, performance |
| Styling | Tailwind CSS | Utility-first, consistent spacing |
| Components | shadcn/ui | Pre-built accessible components |
| Animations | Framer Motion | Page transitions, pyramid build animations |
| Icons | Lucide React | Consistent icon library |
| Diagrams | React Flow | Interactive architecture diagrams |
| Charts | Recharts | Training loss curves, accuracy charts |
| Fonts | Geist (Vercel) | Clean, technical, modern |

### Component Architecture

```
app/
├── layout.tsx              ← Root layout, navbar, footer
├── page.tsx                ← Landing page
├── document-lab/
│   └── page.tsx            ← Document ingestion demo
├── reasoning-lab/
│   └── page.tsx            ← Math reasoning demo
├── architecture/
│   └── page.tsx            ← System design visuals
├── about/
│   └── page.tsx            ← Docs + report

components/
├── ui/                     ← shadcn base components
├── pyramid/
│   ├── PyramidVisual.tsx    ← Animated 4-layer pyramid
│   ├── ChunkExplorer.tsx    ← Side panel for chunk content
│   └── QueryResults.tsx     ← Results display
├── reasoning/
│   ├── ProblemInput.tsx     ← Query textarea + samples
│   ├── StreamingOutput.tsx  ← Token-by-token typewriter
│   └── ModelInfoCard.tsx    ← Model metadata display
├── architecture/
│   ├── SystemDiagram.tsx    ← React Flow diagram
│   └── PipelineFlow.tsx     ← Training pipeline chart
└── shared/
    ├── Navbar.tsx
    ├── Footer.tsx
    └── LoadingStates.tsx
```

### State Management Strategy

- **Local component state** (`useState`) for form inputs and UI toggles
- **Server state** via React Query (TanStack Query) for API calls — handles caching, loading, and error states automatically
- **No global state manager** needed — the app is too small for Redux or Zustand
- **Session data** (uploaded document, current pyramid) stored in React Query cache during the session

---

## 6. Backend Architecture

### Technology: FastAPI (Python)

**Why FastAPI:**

- Same language as the AI pipeline (Python) — no context switching
- Automatic OpenAPI docs generation (`/docs` endpoint) is a bonus for evaluators
- Async support handles concurrent requests efficiently
- Pydantic models enforce request/response validation

### API Structure

```
fastapi_app/
├── main.py                  ← App entry, CORS config, router registration
├── routers/
│   ├── pyramid.py           ← /api/pyramid/* endpoints
│   └── reasoning.py         ← /api/reasoning/* endpoints
├── services/
│   ├── ingestion_service.py ← Wraps Part 1 Python modules
│   └── inference_service.py ← Wraps Part 2 model inference
├── models/
│   ├── request_models.py    ← Pydantic input schemas
│   └── response_models.py   ← Pydantic output schemas
├── core/
│   ├── config.py            ← Environment variables, settings
│   └── session_store.py     ← In-memory or Redis session management
└── utils/
    └── logging.py           ← Structured logging setup
```

### API Endpoints

| Endpoint | Method | Input | Output | Purpose |
|----------|--------|-------|--------|---------|
| `/api/pyramid/ingest` | POST | Document text or file | Pyramid index summary + chunk count | Build pyramid from document |
| `/api/pyramid/query` | POST | Query string + session_id | Matched chunk + level + score | Query the pyramid |
| `/api/pyramid/explore/{level}` | GET | Level (1–4) + session_id | All chunks at that level | Browse pyramid layers |
| `/api/reasoning/solve` | POST | Math problem string | Step-by-step solution + answer | Run reasoning model |
| `/api/reasoning/stream` | POST | Math problem string | Server-Sent Events stream | Streaming token output |
| `/api/health` | GET | — | Status + model loaded | Health check |

### Session Management

- Each document upload creates a `session_id` (UUID)
- Pyramid index for that session stored in Redis (or in-memory dict for single-instance deployment)
- Sessions expire after 30 minutes of inactivity
- This prevents memory buildup without requiring a database

### CORS & Security

- CORS configured to allow only the frontend domain
- Rate limiting: 20 requests/minute per IP using `slowapi`
- File upload size capped at 500KB
- Input text length capped at 50,000 characters

---

## 7. AI Services Architecture

### Service 1 — Pyramid Engine

The Part 1 Python modules (chunker, pyramid builder, retriever) are wrapped as a service class inside the FastAPI app. They do **not** run as a separate process — they are imported and called directly since they are CPU-only.

```
IngestRequest (doc text)
        │
        ▼
IngestionService.ingest(text)
  → calls SlidingWindowChunker
  → calls PyramidBuilder
  → stores result in SessionStore
        │
        ▼
Returns: {session_id, chunk_count, layer_summaries}
```

No GPU needed. Runs entirely on the API server's CPU.

### Service 2 — Reasoning Engine

The fine-tuned model is heavier. Three deployment strategies depending on budget:

**Strategy A — Hosted Inference (Recommended for free tier):**
- Upload fine-tuned LoRA adapter to HuggingFace Hub
- Use HuggingFace Inference API (free tier allows limited calls)
- FastAPI backend calls HuggingFace API and streams response back to frontend
- No GPU needed on your server

**Strategy B — Self-Hosted Model (For paid tier or local demo):**
- Load model at FastAPI startup using `transformers` pipeline
- Run inference in a separate thread to avoid blocking the event loop
- Works on CPU for small models (GPT-2, TinyLLaMA) with ~2–5 second latency
- Works on GPU (Render GPU instance, ~$0.50/hr) for LLaMA 1B with ~1 second latency

**Strategy C — Simulated Demo (If model too heavy):**
- Pre-compute answers for 20 sample problems
- Return pre-computed responses instantly
- Display a clear label: *"Demo responses — full model available locally"*
- This is honest and still demonstrates the system design

---

## 8. Infrastructure & Deployment Architecture

### Zero-Cost Deployment Stack

```
┌─────────────────────────────────────────────────────────┐
│                  PRODUCTION DEPLOYMENT                  │
│                                                         │
│  Frontend: Vercel (Free Tier)                           │
│  ─────────────────────────────                          │
│  • Next.js deploys natively on Vercel                   │
│  • Global CDN, automatic HTTPS                          │
│  • Custom domain support (yourname-vexoo.vercel.app)    │
│                                                         │
│  Backend API: Render (Free Tier)                        │
│  ─────────────────────────────                          │
│  • FastAPI Docker container                             │
│  • Spins down after 15min inactivity (free tier)        │
│  • Auto-redeploys on GitHub push                        │
│                                                         │
│  Session Store: Upstash Redis (Free Tier)               │
│  ─────────────────────────────                          │
│  • 10,000 requests/day free                             │
│  • Serverless Redis — no always-on cost                 │
│                                                         │
│  Model Weights: HuggingFace Hub (Free)                  │
│  ─────────────────────────────                          │
│  • Store LoRA adapter files                             │
│  • Pulled by backend at startup                         │
│                                                         │
│  CDN / Assets: Cloudflare (Free)                        │
│  ─────────────────────────────                          │
│  • Proxy DNS, cache static assets                       │
│  • DDoS protection included                             │
└─────────────────────────────────────────────────────────┘
```

### Deployment Flow

```
Developer pushes to GitHub (main branch)
        │
        ├──── Vercel detects Next.js change
        │         → Builds and deploys frontend automatically
        │         → Live in ~60 seconds
        │
        └──── Render detects Dockerfile change
                  → Builds Docker image
                  → Deploys new backend container
                  → Live in ~3-5 minutes
```

### Docker Strategy for Backend

```
Dockerfile layers:
  1. Base: python:3.11-slim
  2. Install system dependencies
  3. Copy requirements.txt → pip install
  4. Copy AI module code (Part 1 + Part 2)
  5. Copy FastAPI app code
  6. Download model weights at build time (baked in)
     OR pull from HuggingFace at runtime (smaller image)
  7. Expose port 8000
  8. CMD: uvicorn main:app
```

> **Trade-off**: Baking weights into the image makes it large (~2–4GB for LLaMA 1B) but fast to start. Pulling at runtime keeps image small but adds 30–60 second cold start.

> **Recommendation for free tier**: Pull weights at runtime from HuggingFace Hub. Accept the cold start penalty since free tier spins down anyway.

---

## 9. Performance Optimization Plan

### Frontend Performance

| Optimization | Method | Impact |
|-------------|--------|--------|
| Fast initial load | Next.js SSG for static pages (Landing, Architecture, About) | Page loads without waiting for server |
| Code splitting | Next.js automatic per-route splitting | Users only download JS for current page |
| Image optimization | Next.js Image component with lazy loading | Faster rendering, less bandwidth |
| Font loading | `next/font` with `display: swap` | No layout shift during font load |
| Animation performance | Framer Motion `will-change: transform` on animated elements | GPU-accelerated animations, no jank |

### API Performance

| Optimization | Method | Impact |
|-------------|--------|--------|
| Pyramid pre-built | Built at ingestion, not at query time | Query latency <100ms |
| Response caching | Cache identical queries within a session | Instant repeat queries |
| Async endpoints | FastAPI async handlers for I/O operations | No blocking under concurrent load |
| Streaming inference | Server-Sent Events for model output | User sees first token in <1s instead of waiting for full response |
| Render cold start | Ping endpoint every 14 minutes via cron job (UptimeRobot free) | Keeps free tier backend warm |

---

## 10. User Experience Design Principles

### Visual Design Language

- **Theme**: Dark mode by default with a toggle for light mode
- **Color palette**:
  - Background: `#0A0F1E` (deep navy)
  - Surface: `#111827` (dark card)
  - Primary accent: `#3B82F6` (electric blue)
  - Secondary accent: `#8B5CF6` (purple, for AI elements)
  - Success: `#10B981` (green, for correct answers)
  - Text: `#F9FAFB` primary, `#9CA3AF` secondary
- **Typography**: Geist Sans for UI text, Geist Mono for code/model output
- **Border radius**: 8px for cards, 4px for buttons — clean but not overly rounded
- **Spacing**: Consistent 8px grid system throughout

### Interaction Design

- Every user action has an **immediate visual response** (loading spinner, progress bar, or skeleton)
- Pyramid layers animate in sequence during build (L1 first, then L2, L3, L4 — feels like the AI is "thinking")
- Model output streams character by character — not a sudden block of text
- Error states are friendly: *"Something went wrong. Try a sample problem instead."* with a one-click fallback
- Mobile responsive: Document Lab and Reasoning Lab are fully functional on phone screens

### Onboarding Flow

First-time visitors see a subtle tooltip sequence:

1. *"Upload any document or use our sample"*
2. *"Watch the AI build a 4-layer knowledge pyramid"*
3. *"Ask any question — the AI finds the best answer"*

This guides evaluators to the right actions without requiring them to read documentation.

---

## 11. Analytics & Observability

### What to Track (Privacy-Friendly)

- **Page views** per page (no personal data) — via Vercel Analytics (built-in, free)
- **Feature usage**: How many pyramid queries vs reasoning queries per day
- **Error rate**: % of API requests returning 5xx errors
- **Latency**: P50 and P95 response times per endpoint
- **Model inference time**: Logged per request in backend

### Logging Strategy

- Backend uses Python `structlog` for structured JSON logs
- Each request logs: `timestamp`, `endpoint`, `session_id` (hashed), `latency_ms`, `status_code`
- Errors log full traceback to help with debugging
- Logs viewable in Render dashboard (free, 7-day retention)

---

## 12. Repository & Code Organization

### GitHub Repository Structure

```
vexoo-ai-platform/
│
├── frontend/                    ← Next.js application
│   ├── app/                     ← App Router pages
│   ├── components/              ← React components
│   ├── lib/                     ← API client, utilities
│   └── public/                  ← Static assets
│
├── backend/                     ← FastAPI application
│   ├── app/                     ← FastAPI app code
│   ├── ai_modules/              ← Part 1 + Part 2 Python code
│   │   ├── pyramid/             ← Ingestion pipeline modules
│   │   └── reasoning/           ← Training + inference modules
│   ├── Dockerfile               ← Container definition
│   └── requirements.txt
│
├── docs/                        ← Documentation
│   ├── architecture.md          ← This planning document
│   ├── report.pdf               ← One-page assignment summary
│   └── api-reference.md         ← API endpoint documentation
│
├── scripts/                     ← Utility scripts
│   ├── train_gsm8k.py           ← Standalone training script
│   └── evaluate_model.py        ← Standalone eval script
│
├── .github/
│   └── workflows/
│       ├── frontend-deploy.yml  ← Vercel auto-deploy
│       └── backend-deploy.yml   ← Render auto-deploy
│
└── README.md                    ← Professional project README
```

### README Structure (Critical for Impression)

```markdown
# Vexoo Labs AI Platform

> Live Demo: [link] | API Docs: [link] | Report: [PDF link]

## What This Is
[2-sentence description]

## Live Features
- Document Intelligence Lab (Knowledge Pyramid RAG)
- Math Reasoning Lab (GSM8K fine-tuned LLaMA)
- Interactive Architecture Visualization

## Quick Start (Local)
[5 commands max to get running]

## Architecture
[Link to architecture page]

## Assignment Coverage
| Requirement | Status |
|---|---|
| Sliding Window Chunker | ✅ |
| 4-Layer Knowledge Pyramid | ✅ |
| Semantic Query Retrieval | ✅ |
| GSM8K Fine-tuning (LoRA) | ✅ |
| Evaluation Metrics | ✅ |
| Bonus Reasoning Adapter | ✅ |

## Tech Stack
[Brief table]
```

---

## 13. Security Plan

| Layer | Measure | Implementation |
|-------|---------|---------------|
| API | Rate limiting | `slowapi`: 20 req/min per IP |
| File uploads | Size limit + type check | Max 500KB, only `.txt`/`.pdf` accepted |
| Input sanitization | Strip malicious content | Pydantic validators reject unexpected fields |
| CORS | Whitelist frontend domain only | FastAPI CORS middleware |
| HTTPS | Enforced everywhere | Vercel + Render both provide SSL automatically |
| Secrets | Never in code | Environment variables via Vercel + Render dashboards |
| Session isolation | Sessions scoped to session_id | No user can access another user's pyramid |

---

## 14. Launch Checklist

### Before Going Live

- [ ] All 5 pages render correctly on desktop and mobile
- [ ] Document upload and pyramid query work end-to-end
- [ ] At least 5 sample math problems return correct answers
- [ ] Architecture diagrams are accurate and interactive
- [ ] PDF report is downloadable from About page
- [ ] GitHub repo is public and README is polished
- [ ] Custom domain or clean Vercel URL configured
- [ ] Health check endpoint returns 200
- [ ] Error states display friendly messages (not raw stack traces)
- [ ] Backend warms up within 30 seconds of first request

### What to Send to Vexoo Labs

Instead of just a zip file, send:

```
Subject: Vexoo Labs Assignment — [Your Name]

Hi team,

Submission includes:

  Live Platform:     https://yourname-vexoo.vercel.app
  GitHub Repo:       https://github.com/yourname/vexoo-ai-platform
  Assignment Report: [attached PDF]
  API Documentation: https://yourname-vexoo.vercel.app/docs

The platform demonstrates both deliverables live in browser —
no setup required to evaluate.

ZIP file attached as backup per submission requirements.
```

---

## 15. Future Enhancements (Post-Submission)

- **Authentication**: Add GitHub OAuth so users can save their pyramid sessions
- **Multi-document support**: Allow uploading multiple documents and querying across all of them
- **Real embeddings**: Replace TF-IDF with sentence-transformers for dramatically better retrieval
- **Model comparison**: Side-by-side comparison of base LLaMA vs fine-tuned LLaMA on the same problem
- **Export feature**: Allow users to download the full pyramid structure as JSON
- **Evaluation dashboard**: Live chart showing model accuracy across question difficulty levels
- **Reasoning adapter demo**: Interactive UI showing the bonus adapter routing different question types
