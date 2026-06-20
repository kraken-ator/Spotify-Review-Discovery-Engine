# 🎧 Spotify Review Discovery Engine

> A live AI-powered review intelligence engine built for the **NextLeap PM Fellowship** — surfacing why users struggle to discover new music on Spotify, straight from the words of real users.

---

## 📋 Project Overview

This is a **live review intelligence engine**, not a static analysis. It runs an end-to-end pipeline that ingests, clusters, and synthesizes real user feedback into product-ready insight.

**The full workflow, start to finish:**

1. **Ingest** — The engine pulls reviews and posts from five sources: the **App Store** and **Play Store** (scraped live, no credentials needed), **Spotify Community Forums** (scraped live), and **Reddit** and **Twitter/X** (loaded via Apify JSON exports, since both platforms' live APIs are either paywalled or heavily rate-limited on free tiers).
2. **Filter for relevance** — Every review is checked against a discovery-specific keyword set (terms like *"recommendation," "algorithm," "discover weekly," "repetitive," "same songs"*) before it enters the pipeline. This keeps the dataset focused on *music discovery* feedback rather than unrelated complaints (bugs, pricing, ads).
3. **Embed & cluster semantically** — Relevant reviews are converted into sentence embeddings and grouped using **HDBSCAN**, which discovers the natural number of clusters in the data itself — it does not force a fixed number of categories.
4. **Synthesize with an LLM** — Each cluster is sent to **Groq's LLM API**, which names the pattern, identifies its root cause, infers the underlying user emotion, and writes a PM-ready insight — producing a set of data-driven **"Failure Modes."**
5. **Present** — Everything renders in an interactive **Streamlit dashboard**: the failure modes, a semantic cluster map, source/volume breakdowns, and an auto-generated JTBD interview guide for further research.

---

## ✨ Key Features

- **🔄 Automated multi-source ingestion** — App Store, Play Store, Spotify Community, Reddit, and Twitter/X, unified into a single relevance-filtered dataset with zero manual collation.
- **🧩 Dynamic clustering, not forced categories** — Cluster count is *discovered* via HDBSCAN (silhouette-selected KMeans as a fallback), so the number of Failure Modes reflects what the data actually contains.
- **🤖 LLM-powered synthesis** — Groq turns raw clusters into named failure modes with root cause, emotional signature, business impact, and a PM insight for each.
- **🛡️ Fail-safe by design** — If live synthesis fails, the engine falls back to cached content automatically — and **flags it as a fallback** rather than presenting it as live output.
- **📊 Interactive dashboard** — Explore failure modes, a semantic review map, and source coverage, all in one Streamlit interface.

---

## 🧪 Reviewer Testing Guide

You have two ways to evaluate this project — pick whichever fits your time.

### ✅ Option 1 (Recommended): Zero-Click Cloud Dashboard

Click the **Streamlit Cloud link in the presentation deck**. That's it.

The environment, ML libraries, and API keys are fully hosted — there is nothing to install and nothing to configure. The dashboard is live and ready to interact with directly in your browser.

### 🛠️ Option 2: Testing the Backend Locally

If you'd like to inspect the backend pipeline directly:

```bash
# 1. Clone the repo and enter the project folder
git clone <repo-url>
cd Spotify-Review-Discovery-Engine

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run either of the following:

# Option A — run the raw backend orchestrator (ingestion → clustering → synthesis)
python run_pipeline.py

# Option B — launch the full interactive UI
streamlit run app.py
```

> 💡 For live LLM synthesis locally, copy `env.example` to `.env` and add a free [Groq API key](https://console.groq.com). Without it, the app runs in demo mode automatically — see the limitations section below.

---

## ⚠️ API Limitations (Important)

This engine runs on a **free-tier Groq API key**, which is subject to strict rate limits (requests per minute and per day).

If that limit is hit during live testing, the engine is built to **fail gracefully**: it automatically falls back to cached demo-mode data so the dashboard **never crashes**. Any fallback content is explicitly flagged in the UI — it is never silently presented as live synthesis, and real review counts are never merged with placeholder text without disclosure.

If you see a fallback notice while testing, it simply means the rate limit was hit — the underlying pipeline and dashboard are still fully functional.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **App / UI** | Python, Streamlit |
| **Embeddings & Clustering** | Sentence-Transformers, HDBSCAN, UMAP (PCA/KMeans fallback) |
| **LLM Synthesis** | Groq |
| **Scraping & Ingestion** | Requests, BeautifulSoup, google-play-scraper, Apify exports |

---

## 📁 Project Structure

```
Spotify-Review-Discovery-Engine/
├── app.py                  # Streamlit dashboard — the main UI
├── run_pipeline.py         # CLI orchestrator: ingest → cluster → synthesize
├── requirements.txt
├── env.example             # Template for GROQ_API_KEY and optional sources
├── src/
│   ├── scraper.py          # Multi-source ingestion + discovery-relevance filtering
│   ├── clusterer.py        # Embedding + dynamic semantic clustering
│   └── synthesizer.py      # Groq LLM synthesis → Failure Modes + JTBD guide
└── data/
    ├── failure_modes.json  # Latest synthesized output
    ├── clusters.json       # Latest cluster metadata
    └── apify_*.json        # Reddit / Twitter exports (optional, for offline runs)
```
