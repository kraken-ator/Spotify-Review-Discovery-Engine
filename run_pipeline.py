"""
run_pipeline.py — Full Review Engine Orchestrator
Run this once before launching the Streamlit app to pre-compute all data.

Usage:
  python run_pipeline.py                          # demo mode only — no scraping, no keys needed
  python run_pipeline.py --live                    # scrape App Store + Play Store + Community (free, keyword-filtered)
  python run_pipeline.py --live --reddit           # also load Apify Reddit JSON export
  python run_pipeline.py --live --twitter-live     # also attempt live Twitter API (needs paid tier; opt-in, NOT bundled into --live)
  python run_pipeline.py --live --include-seed     # also blend in 25 synthetic seed reviews (disclosed in output, not recommended for a real submission)
  python run_pipeline.py --skip-scrape             # re-synthesize from existing raw_reviews.csv
  python run_pipeline.py --apify-file data/my_reddit_export.json   # custom Apify Reddit file path

Cluster count is discovered from the data (HDBSCAN, or silhouette-selected
KMeans if hdbscan isn't installed) — there is no --target-clusters flag
because forcing a predetermined count was exactly the issue this version
fixes. Use --min-cluster-size / --min-cluster-pct / --max-clusters to tune
the discovery process, not to dictate its outcome.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent / "src"))


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║         Spotify · PM Fellowship · Review Intelligence Engine     ║
║         Sources: App Store · Play Store · Community (free)       ║
║                  Reddit (Apify) · Twitter (Apify or live, opt-in)║
║         LLM: Groq                                                ║
╚══════════════════════════════════════════════════════════════════╝
""")


def validate_env(require_llm: bool = False, need_twitter_live: bool = False):
    """Check environment variables, warn on missing keys."""
    from dotenv import load_dotenv
    load_dotenv()

    groq_key = os.getenv("GROQ_API_KEY", "")
    twitter_key = os.getenv("TWITTER_BEARER_TOKEN", "")

    if require_llm and (not groq_key or groq_key == "your-groq-api-key-here"):
        log.warning("⚠️  GROQ_API_KEY not set — will use demo failure modes instead of live synthesis")

    if need_twitter_live and (not twitter_key or twitter_key == "your-twitter-bearer-token-here"):
        log.warning(
            "⚠️  --twitter-live was passed but TWITTER_BEARER_TOKEN is not set. "
            "Note: free-tier Twitter API keys return 402 (paywalled) for search — "
            "this will likely fail even with a token set."
        )

    return groq_key, twitter_key


def run(args):
    print_banner()

    groq_key, _ = validate_env(require_llm=args.live, need_twitter_live=args.twitter_live)
    use_live_synthesis = (
        args.live
        and bool(groq_key)
        and groq_key != "your-groq-api-key-here"
    )

    Path("data").mkdir(exist_ok=True)

    # ── STEP 1: Ingest ──────────────────────────────────────────────────────
    if not args.skip_scrape:
        log.info("STEP 1/3 — Ingesting reviews")
        from scraper import ingest_all_reviews

        df = ingest_all_reviews(
            use_seed=args.include_seed,
            use_appstore=args.live,
            use_playstore=args.live,
            use_reddit_apify=args.reddit,
            use_twitter_apify=args.twitter_apify,
            use_community=args.live,
            # Twitter's live API is decoupled from --live: it returns 402
            # (paywalled) on the free tier and was previously attempted on
            # every --live run regardless, wasting time on a call that
            # will not succeed without a paid plan. Opt in explicitly.
            use_twitter=args.twitter_live,
            filter_discovery=not args.no_discovery_filter,
            max_per_source=args.max_reviews,
            apify_reddit_file=args.apify_file,
            output_path="data/raw_reviews.csv",
        )
        if len(df) == 0:
            log.error(
                "No reviews collected. Pass --live to enable scraping (App Store, "
                "Play Store, Community need no credentials), or --include-seed for "
                "an offline demo run."
            )
            sys.exit(1)
        n_seed = int((df["provenance"] == "synthetic_seed").sum()) if "provenance" in df.columns else 0
        n_scraped = len(df) - n_seed
        log.info(f"  ✓ {len(df)} reviews ingested ({n_scraped} scraped + {n_seed} synthetic seed) → data/raw_reviews.csv")

    else:
        import pandas as pd
        if not Path("data/raw_reviews.csv").exists():
            log.error("data/raw_reviews.csv not found. Remove --skip-scrape or run without it first.")
            sys.exit(1)
        df = pd.read_csv("data/raw_reviews.csv")
        n_seed = int((df["provenance"] == "synthetic_seed").sum()) if "provenance" in df.columns else 0
        n_scraped = len(df) - n_seed
        log.info(f"  ✓ Loaded {len(df)} existing reviews from data/raw_reviews.csv ({n_scraped} scraped + {n_seed} synthetic seed)")

    # ── STEP 2: Cluster ─────────────────────────────────────────────────────
    # Cluster count is discovered from the data — no forced k. See
    # clusterer.py's cluster_reviews() / consolidate_small_clusters().
    log.info("STEP 2/3 — Embedding & clustering (natural cluster count, not forced)")
    try:
        from clusterer import run_clustering_pipeline
        clusters_meta, df_clustered = run_clustering_pipeline(
            df,
            output_path="data/clusters.json",
            min_cluster_size=args.min_cluster_size,
            min_cluster_pct=args.min_cluster_pct,
            max_clusters=args.max_clusters,
        )
        log.info(f"  ✓ {len(clusters_meta)} clusters discovered → data/clusters.json")

    except Exception as e:
        log.error(f"Clustering failed: {e}")
        log.warning("Using single-cluster fallback for synthesis")
        clusters_meta = {
            0: {
                "cluster_id": 0,
                "size": len(df),
                "pct_of_total": 100.0,
                "avg_rating": df["rating"].dropna().mean() if "rating" in df else None,
                "representative_quotes": [
                    {"text": row["text"][:250], "source": row.get("source", "unknown"),
                     "rating": row.get("rating"), "helpful_votes": int(row.get("helpful_votes", 0))}
                    for _, row in df.head(6).iterrows()
                ],
                "sources": df["source"].value_counts().to_dict() if "source" in df else {},
            }
        }

    # ── STEP 3: Synthesize ──────────────────────────────────────────────────
    log.info("STEP 3/3 — Synthesizing failure modes")

    if use_live_synthesis:
        log.info("  Using live Groq synthesis (GROQ_API_KEY set)")
        from synthesizer import run_synthesis_pipeline
        result = run_synthesis_pipeline(
            clusters_meta,
            output_path="data/failure_modes.json",
            generate_guide=True,
        )
    else:
        log.info("  Using pre-analyzed demo-mode failure modes (set GROQ_API_KEY for live synthesis)")
        from synthesizer import DEMO_FAILURE_MODES
        import copy
        result = copy.deepcopy(DEMO_FAILURE_MODES)
        # IMPORTANT: do NOT overwrite the demo dataset's own honest
        # total_reviews_analyzed (25, the actual seed count it represents)
        # with this run's real len(df). That exact substitution — a real
        # count paired with fabricated qualitative content, displayed with
        # no flag — was the most damaging issue found in review. Instead,
        # record this run's real counts under separate, clearly-named
        # keys so nothing is silently merged into the demo content's own
        # figure.
        result["metadata"]["n_reviews_scraped"] = n_scraped
        result["metadata"]["n_reviews_synthetic_seed_this_run"] = n_seed
        result["metadata"]["note"] = (
            f"This run collected {len(df)} real reviews, but the failure mode "
            f"descriptions below are demo-mode placeholder content (no GROQ_API_KEY "
            f"was set), not synthesized from those {len(df)} reviews. Set "
            f"GROQ_API_KEY for genuine synthesis."
        )
        with open("data/failure_modes.json", "w") as f:
            json.dump(result, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────────
    fms    = result.get("failure_modes", [])
    master = result.get("master_synthesis", {})
    meta   = result.get("metadata", {})
    n_fallback = sum(1 for fm in fms if fm.get("synthesis_status", "live") != "live")

    print(f"""
{'='*68}
  ENGINE COMPLETE
{'='*68}
  Reviews ingested  : {len(df)} ({n_scraped} scraped + {n_seed} synthetic seed)
  Failure modes     : {len(fms)} (data-driven count, not forced)
  LLM               : {meta.get('llm_provider', 'Groq')}
  Synthesis mode    : {'Live (Groq)' if use_live_synthesis else 'Demo mode (no API key)'}
  Fallback cards    : {n_fallback}/{len(fms)} {'⚠ check synthesis_status before presenting this run' if n_fallback else '— all live'}

  Master Insight:
  "{master.get('master_insight_title', 'N/A')}"

  Failure Modes Identified:
""")
    for i, fm in enumerate(fms):
        impact = fm.get("business_impact", "?")
        pct    = fm.get("pct_of_total", "?")
        flag   = " ⚠ FALLBACK" if fm.get("synthesis_status", "live") != "live" else ""
        print(f"  {i+1}. {fm.get('failure_mode_name', 'N/A')} ({pct}% of reviews, {impact} impact){flag}")

    print(f"""
  Output files:
    data/raw_reviews.csv          — all ingested reviews (provenance column discloses scraped vs. synthetic)
    data/clusters.json            — cluster metadata
    data/clusters_reviews.csv     — reviews with cluster assignments (umap_x/umap_y for the scatter plot)
    data/failure_modes.json       — full synthesis output (synthesis_status on every object)

  Next step → launch the app:
    streamlit run app.py
{'='*68}
""")

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Spotify Review Intelligence Engine Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Scrape App Store, Play Store, and Spotify Community (all free, no credentials needed); use live Groq synthesis"
    )
    parser.add_argument(
        "--reddit", action="store_true",
        help="Load Reddit data from Apify JSON export (see APIFY_REDDIT_FILE in .env)"
    )
    parser.add_argument(
        "--twitter-apify", action="store_true",
        help="Load Twitter data from Apify JSON export"
    )
    parser.add_argument(
        "--twitter-live", action="store_true",
        help="Attempt the live Twitter API (separate from --live — free-tier keys return 402/paywalled, "
             "so this is opt-in rather than bundled into every --live run)"
    )
    parser.add_argument(
        "--include-seed", action="store_true",
        help="Blend in 25 synthetic seed reviews (off by default). Every review is tagged with a "
             "`provenance` column either way, but this content should not be presented as real "
             "review data without disclosure — see scraper.py's SEED_REVIEWS."
    )
    parser.add_argument(
        "--apify-file", type=str, default=None,
        metavar="PATH",
        help="Path to Apify Reddit JSON export (overrides APIFY_REDDIT_FILE in .env)"
    )
    parser.add_argument(
        "--skip-scrape", action="store_true",
        help="Skip ingestion and use existing data/raw_reviews.csv"
    )
    parser.add_argument(
        "--max-reviews", type=int, default=500,
        help="Max reviews per source after discovery-relevance filtering (default: 500)"
    )
    parser.add_argument(
        "--no-discovery-filter", action="store_true",
        help="Disable the App Store / Play Store keyword relevance filter (for debugging/comparison only — "
             "without it, reviews are dominated by ads/bugs/pricing complaints, not discovery signal)"
    )
    parser.add_argument(
        "--min-cluster-size", type=int, default=15,
        help="HDBSCAN min_cluster_size — minimum points to form a cluster (default: 15)"
    )
    parser.add_argument(
        "--min-cluster-pct", type=float, default=3.0,
        help="Clusters below this %% of total reviews get merged into their nearest larger cluster (default: 3.0)"
    )
    parser.add_argument(
        "--max-clusters", type=int, default=8,
        help="Presentation backstop: if more than this many clusters survive consolidation, "
             "merge the smallest until at or below this cap (default: 8)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
