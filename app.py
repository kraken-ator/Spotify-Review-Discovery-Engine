"""
app.py — Spotify Discovery Review Intelligence Engine
Streamlit app: ingest → cluster → synthesize → display N failure modes
(N is discovered by clusterer.py from the data, not fixed at 5)
"""

import os
import json
import time
import logging
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from dotenv import load_dotenv

import sys
sys.path.insert(0, "src")
load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="Spotify · Discovery Intelligence Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Design tokens ──────────────────────────────────────────────────────────────
SPOTIFY_GREEN = "#1DB954"
DARK_BG       = "#121212"
CARD_BG       = "#1E1E1E"
CARD_BORDER   = "#2A2A2A"
TEXT_PRIMARY  = "#FFFFFF"
TEXT_MUTED    = "#B0B0B0"   # bumped from A0 → B0 for better visibility
TEXT_DIM      = "#888888"   # bumped from 6B → 88
FALLBACK_AMBER = "#FFB800"

IMPACT_COLORS = {"HIGH": "#FF4444", "MEDIUM": "#FFB800", "LOW": "#1DB954"}
FRICTION_COLORS = {
    "algorithmic": "#7C3AED",
    "ux":          "#2563EB",
    "contextual":  "#059669",
    "social":      "#D97706",
    "cognitive":   "#DB2777",
}
FRICTION_LABELS = {
    "algorithmic": "Algorithmic",
    "ux":          "UX / Design",
    "contextual":  "Context-Blind",
    "social":      "Social",
    "cognitive":   "Cognitive Load",
}
# Extended to 8 — clusterer.py now discovers a natural cluster count
# (capped at max_clusters=8 by default) rather than always producing
# exactly 5, so the palette needs headroom beyond 5 entries.
CLUSTER_ACCENT_COLORS = ["#1DB954", "#7C3AED", "#2563EB", "#DB2777", "#FFB800", "#06B6D4", "#F97316", "#84CC16"]

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
  }}
  .stApp {{ background-color: {DARK_BG}; }}

  /* ── Hero ── */
  .hero {{
    background: linear-gradient(135deg, #0A0A0A 0%, #1A1A1A 50%, #0D1F0D 100%);
    border: 1px solid {CARD_BORDER};
    border-radius: 16px;
    padding: 48px 40px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
  }}
  .hero::before {{
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(29,185,84,0.12) 0%, transparent 70%);
    border-radius: 50%;
  }}
  .hero-eyebrow {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: {SPOTIFY_GREEN}; margin-bottom: 12px;
  }}
  .hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 32px; font-weight: 700; line-height: 1.15;
    color: {TEXT_PRIMARY}; margin-bottom: 12px;
  }}
  .hero-subtitle {{
    font-size: 15px; color: {TEXT_MUTED};
    line-height: 1.6; max-width: 680px; margin-bottom: 0;
  }}
  .stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 28px; }}
  .stat-pill {{
    background: rgba(29,185,84,0.1);
    border: 1px solid rgba(29,185,84,0.25);
    border-radius: 100px; padding: 6px 16px;
    font-size: 13px; font-weight: 500; color: {SPOTIFY_GREEN};
  }}

  /* ── Section headers — BRIGHTER ── */
  .section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 12px; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #AAAAAA;                /* was TEXT_DIM (#6B6B6B) */
    margin: 36px 0 16px;
    padding-bottom: 8px;
    border-bottom: 1px solid #3A3A3A;
  }}

  /* ── Synthesis box ── */
  .synthesis-box {{
    background: linear-gradient(135deg, #0D1F0D, #151515);
    border: 1px solid rgba(29,185,84,0.3);
    border-radius: 12px; padding: 32px; margin: 24px 0;
  }}
  .synthesis-box-flagged {{
    background: linear-gradient(135deg, #1E1A0C, #151515) !important;
    border: 1px dashed rgba(255,184,0,0.5) !important;
  }}
  .synthesis-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: {SPOTIFY_GREEN}; margin-bottom: 14px; line-height: 1.3;
  }}
  .synthesis-text {{
    font-size: 14px; color: #E0E0E0;
    line-height: 1.7; margin-bottom: 16px;
  }}
  /* ── BRIGHTER sub-labels inside synthesis box ── */
  .synthesis-label {{
    font-size: 11px; font-weight: 700;
    letter-spacing: 1.8px; text-transform: uppercase;
    color: #CCCCCC;                /* was TEXT_DIM */
    margin-bottom: 6px; margin-top: 4px;
  }}

  /* ── Failure mode cards ── */
  .fm-card {{
    background: {CARD_BG}; border: 1px solid {CARD_BORDER};
    border-radius: 12px; padding: 28px; margin-bottom: 16px;
    position: relative; transition: border-color 0.2s;
  }}
  .fm-card:hover {{ border-color: #3A3A3A; }}
  .fm-accent-bar {{
    position: absolute; left: 0; top: 16px; bottom: 16px;
    width: 3px; border-radius: 0 2px 2px 0;
  }}
  .fm-number {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px; font-weight: 700; letter-spacing: 1.5px;
    color: #999999; text-transform: uppercase; margin-bottom: 4px;
  }}
  .fm-name {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 20px; font-weight: 700;
    color: {TEXT_PRIMARY}; margin-bottom: 6px; line-height: 1.2;
  }}
  .fm-tagline {{
    font-size: 14px; color: {TEXT_MUTED};
    font-style: italic; margin-bottom: 16px; line-height: 1.5;
  }}
  .fm-meta-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }}
  .badge {{
    display: inline-block; padding: 3px 10px;
    border-radius: 100px; font-size: 11px;
    font-weight: 600; letter-spacing: 0.5px;
  }}
  /* ── BRIGHTER card sub-labels ── */
  .fm-section-label {{
    font-size: 11px; font-weight: 700;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: #BBBBBB;                /* was TEXT_DIM */
    margin-bottom: 6px;
  }}
  .fm-root-cause {{
    font-size: 13.5px; color: #D8D8D8; line-height: 1.6;
    margin-bottom: 16px; padding: 12px 14px;
    background: rgba(255,255,255,0.03);
    border-radius: 8px; border-left: 2px solid #444;
  }}
  .fm-pm-insight {{
    font-size: 13px; color: {SPOTIFY_GREEN}; line-height: 1.6;
    margin-bottom: 16px; padding: 10px 14px;
    background: rgba(29,185,84,0.06);
    border-radius: 8px; border-left: 2px solid {SPOTIFY_GREEN};
  }}
  .fm-jtbd {{
    font-size: 13px; color: #5b9df5; line-height: 1.6;
    margin-bottom: 4px; padding: 10px 14px;
    background: rgba(91,157,245,0.06);
    border-radius: 8px; border-left: 2px solid #5b9df5;
  }}

  /* ── Fallback / demo-mode disclosure ──
       A card showing this content must be visually distinct from one
       with genuine live synthesis — this is the fix for the silent
       fallback-content issue: a failed synthesis can no longer render
       identically to a real one. ── */
  .fm-card-flagged {{
    border: 1px dashed {FALLBACK_AMBER}66 !important;
    background: linear-gradient(135deg, {CARD_BG}, #1E1A0C) !important;
  }}
  .fallback-banner {{
    background: rgba(255,184,0,0.12);
    border: 1px solid rgba(255,184,0,0.4);
    border-radius: 8px; padding: 10px 14px; margin-bottom: 18px;
    font-size: 12.5px; color: {FALLBACK_AMBER}; line-height: 1.5;
    font-weight: 600;
  }}

  /* ── Quote cards ── */
  .quote-block {{
    background: rgba(255,255,255,0.025); border: 1px solid {CARD_BORDER};
    border-radius: 8px; padding: 14px 16px; margin-bottom: 8px;
    font-size: 13px; color: #C8C8C8; line-height: 1.6; font-style: italic;
  }}
  .quote-meta {{
    font-size: 11px; color: #888; margin-top: 6px; font-style: normal;
  }}

  /* ── Coverage summary cards ── */
  .coverage-card {{
    background: {CARD_BG}; border: 1px solid {CARD_BORDER};
    border-radius: 10px; padding: 16px 20px;
  }}
  .coverage-pct {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px; font-weight: 700; line-height: 1;
  }}
  .coverage-name {{
    font-size: 12px; color: #BBBBBB;
    margin-top: 4px; line-height: 1.3;
  }}
  .coverage-meta {{
    font-size: 11px; color: #888; margin-top: 3px;
  }}

  /* ── Interview guide ── */
  .interview-card {{
    background: {CARD_BG}; border: 1px solid {CARD_BORDER};
    border-radius: 10px; padding: 20px 24px; margin-bottom: 12px;
  }}
  .interview-q {{
    font-size: 15px; font-weight: 600; color: {TEXT_PRIMARY};
    margin-bottom: 8px; line-height: 1.4;
  }}
  .interview-probe {{
    font-size: 13px; color: {TEXT_MUTED};
    padding-left: 12px; border-left: 2px solid {CARD_BORDER};
    margin-top: 6px; line-height: 1.5;
  }}

  /* ── Run button ── */
  .stButton > button {{
    background: {SPOTIFY_GREEN} !important; color: #000 !important;
    font-weight: 700 !important; font-size: 14px !important;
    border: none !important; border-radius: 100px !important;
    padding: 10px 28px !important; letter-spacing: 0.3px !important;
  }}
  .stButton > button:hover {{ background: #1ed760 !important; }}

  .divider {{ height: 1px; background: {CARD_BORDER}; margin: 28px 0; }}
  div[data-testid="stExpander"] {{
    background: {CARD_BG} !important;
    border: 1px solid {CARD_BORDER} !important;
    border-radius: 10px !important;
  }}
  .stProgress > div > div {{ background-color: {SPOTIFY_GREEN} !important; }}
  footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Data loaders ───────────────────────────────────────────────────────────────

def load_failure_modes():
    path = Path("data/failure_modes.json")
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def load_clustered_reviews():
    path = Path("data/clusters_reviews.csv")
    if path.exists():
        return pd.read_csv(path)
    return None

# ── Pipeline runner ────────────────────────────────────────────────────────────

def run_full_pipeline(use_seed=False, use_appstore=True, use_playstore=True,
                      use_reddit=False, use_community=True, use_twitter=False,
                      use_twitter_apify=False, filter_discovery=True,
                      apify_reddit_path=None, apify_twitter_path=None):
    import sys
    sys.path.insert(0, "src")
    from scraper import ingest_all_reviews
    from clusterer import run_clustering_pipeline
    from synthesizer import run_synthesis_pipeline, DEMO_FAILURE_MODES

    progress = st.progress(0)
    status = st.empty()

    status.markdown(f"<p style='color:{TEXT_MUTED};font-size:13px'>⟳ Ingesting reviews from selected sources...</p>", unsafe_allow_html=True)
    df = ingest_all_reviews(
        use_appstore=use_appstore, use_playstore=use_playstore,
        use_reddit_apify=use_reddit, use_community=use_community,
        use_twitter=use_twitter, use_twitter_apify=use_twitter_apify,
        use_seed=use_seed, filter_discovery=filter_discovery,
        apify_reddit_file=apify_reddit_path,
        apify_twitter_file=apify_twitter_path or "data/apify_twitter.json",
        output_path="data/raw_reviews.csv",
    )

    if df is None or len(df) == 0:
        progress.empty()
        status.markdown(
            f"<p style='color:#FF4444;font-size:13px'>⚠ No reviews were collected from the selected "
            f"sources. Turn on at least one data source (App Store, Play Store, and Community Forums "
            f"need no credentials), or enable synthetic seed reviews for an offline demo.</p>",
            unsafe_allow_html=True,
        )
        return None

    n_seed = int((df["provenance"] == "synthetic_seed").sum()) if "provenance" in df.columns else 0
    n_scraped = len(df) - n_seed
    progress.progress(33)

    status.markdown(f"<p style='color:{TEXT_MUTED};font-size:13px'>⟳ Embedding & clustering {len(df)} reviews ({n_scraped} scraped, {n_seed} synthetic)...</p>", unsafe_allow_html=True)
    try:
        clusters_meta, _ = run_clustering_pipeline(df, output_path="data/clusters.json")
        progress.progress(66)
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key and api_key != "your-groq-api-key-here":
            status.markdown(f"<p style='color:{TEXT_MUTED};font-size:13px'>⟳ Synthesizing failure modes with Groq...</p>", unsafe_allow_html=True)
            result = run_synthesis_pipeline(clusters_meta, output_path="data/failure_modes.json")
        else:
            status.markdown(f"<p style='color:{FALLBACK_AMBER};font-size:13px'>⚠ No GROQ_API_KEY set — using demo-mode content, clearly flagged on every card.</p>", unsafe_allow_html=True)
            result = DEMO_FAILURE_MODES
            os.makedirs("data", exist_ok=True)
            with open("data/failure_modes.json", "w") as f:
                json.dump(result, f, indent=2)
    except Exception as e:
        status.markdown(f"<p style='color:#FF4444;font-size:13px'>Error: {e}</p>", unsafe_allow_html=True)
        from synthesizer import DEMO_FAILURE_MODES
        result = DEMO_FAILURE_MODES
        os.makedirs("data", exist_ok=True)
        with open("data/failure_modes.json", "w") as f:
            json.dump(result, f, indent=2)

    # Disclose seed/scraped provenance breakdown alongside the rest of the
    # metadata so it survives into the dashboard, regardless of whether
    # this run hit the live-synthesis or demo-mode branch above.
    if isinstance(result, dict):
        result.setdefault("metadata", {})
        result["metadata"]["n_reviews_scraped"] = n_scraped
        result["metadata"]["n_reviews_synthetic_seed"] = n_seed

    progress.progress(100)
    status.empty()
    progress.empty()
    return result


# ── Visualizations ─────────────────────────────────────────────────────────────

def render_cluster_scatter(df_clustered, failure_modes):
    if "umap_x" not in df_clustered.columns:
        return
    name_map = {fm["cluster_id"]: fm["failure_mode_name"] for fm in failure_modes}
    df_clustered = df_clustered.copy()
    df_clustered["failure_mode"] = df_clustered["cluster_id"].map(name_map).fillna("Uncategorized")
    df_clustered["text_preview"] = df_clustered["text"].astype(str).str[:80] + "..."

    fig = px.scatter(
        df_clustered, x="umap_x", y="umap_y",
        color="failure_mode",
        color_discrete_sequence=CLUSTER_ACCENT_COLORS,
        hover_data={"text_preview": True, "umap_x": False, "umap_y": False, "source": True},
        labels={"failure_mode": "Failure Mode"},
    )
    fig.update_traces(marker=dict(size=7, opacity=0.85))
    fig.update_layout(
        paper_bgcolor="#161616", plot_bgcolor="#161616",
        font=dict(color="#CCCCCC", family="Inter"),
        legend=dict(bgcolor=CARD_BG, bordercolor="#444", borderwidth=1,
                    font=dict(size=12, color="#FFFFFF")),
        xaxis=dict(visible=True, 
    showgrid=True, 
    gridcolor="#2A2A2A", 
    zeroline=False, 
    title="Semantic Dimension 1 (UMAP)",
    title_font=dict(size=12, color="#888888"),
    tickfont=dict(color="#555555")),
        yaxis=dict(visible=True, 
    showgrid=True, 
    gridcolor="#2A2A2A", 
    zeroline=False, 
    title="Semantic Dimension 2 (UMAP)",
    title_font=dict(size=12, color="#888888"),
    tickfont=dict(color="#555555")),
        margin=dict(l=0, r=0, t=10, b=0),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_impact_chart(failure_modes):
    names   = [fm["failure_mode_name"] for fm in failure_modes]
    sizes   = [fm.get("review_count", 0) for fm in failure_modes]
    impacts = [fm.get("business_impact", "MEDIUM") for fm in failure_modes]
    colors  = [IMPACT_COLORS.get(i, "#888") for i in impacts]

    fig = go.Figure(go.Bar(
        x=sizes, y=names, orientation="h",
        marker_color=colors,
        text=[f"{s} reviews — {i} impact" for s, i in zip(sizes, impacts)],
        textposition="inside",
        textfont=dict(color="white", size=12, family="Inter"),
    ))
    fig.update_layout(
        paper_bgcolor=DARK_BG, plot_bgcolor="#161616",
        font=dict(color="#CCCCCC", family="Inter"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
        yaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=12, color="#FFFFFF")),
        margin=dict(l=0, r=0, t=0, b=0),
        height=max(200, len(failure_modes) * 52),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Failure mode card ──────────────────────────────────────────────────────────

def render_failure_mode_card(fm, idx):
    accent        = CLUSTER_ACCENT_COLORS[idx % len(CLUSTER_ACCENT_COLORS)]
    impact        = fm.get("business_impact", "MEDIUM")
    impact_score = fm.get("impact_score", 0)
    impact_color  = IMPACT_COLORS.get(impact, "#888")
    friction_type = fm.get("discovery_friction_type", "algorithmic")
    friction_color = FRICTION_COLORS.get(friction_type, "#888")
    friction_label = FRICTION_LABELS.get(friction_type, friction_type.title())

    # ── Fallback / demo-mode disclosure ──
    # synthesis_status is "live" for genuine LLM synthesis, or one of
    # "fallback_demo" / "fallback_generic" / "demo_mode" when live
    # synthesis failed (or wasn't attempted at all, in full demo mode).
    # This card must never look identical to a genuinely-synthesized one
    # in those cases — that silent merge was the most damaging issue
    # found in review.
    status = fm.get("synthesis_status", "live")
    is_flagged = status != "live"
    card_class = "fm-card fm-card-flagged" if is_flagged else "fm-card"

    banner_html = ""
    if is_flagged:
        banner_text = {
            "fallback_demo": "⚠️ FALLBACK CONTENT — Live LLM synthesis failed for this cluster after retry. The qualitative description below is placeholder content paired with this run's real review count, not genuine synthesis of these specific reviews.",
            "fallback_generic": "⚠️ SYNTHESIS PENDING — Live LLM synthesis failed and no fallback content exists for this cluster. Re-run the engine to attempt synthesis again.",
            "demo_mode": "ℹ️ DEMO MODE — Illustrative content from the 25 synthetic seed reviews, not generated from a live pipeline run.",
        }.get(status, "⚠️ FALLBACK CONTENT — not genuine live synthesis.")
        banner_html = f'<div class="fallback-banner">{banner_text}</div>'

    jtbd_html = ""
    if fm.get("jobs_to_be_done"):
        jtbd_html = f'<div class="fm-section-label">Job To Be Done (retrospective)</div><div class="fm-jtbd">🎯 {fm.get("jobs_to_be_done", "")}</div>'

    st.markdown(f"""
    <div class="{card_class}">
      <div class="fm-accent-bar" style="background:{accent}"></div>
      {banner_html}
      <div class="fm-number">Failure Mode {idx + 1} · {fm.get('pct_of_total', '?')}% of reviews</div>
      <div class="fm-name">{fm.get('failure_mode_name', 'Unnamed')}</div>
      <div class="fm-tagline">"{fm.get('tagline', '')}"</div>
      <div class="fm-meta-row">
        <span class="badge" style="background:{impact_color}22;color:{impact_color};border:1px solid {impact_color}55">
          {impact} business impact
        </span>
        <span class="badge" style="background:{friction_color}22;color:{friction_color};border:1px solid {friction_color}55">
          {friction_label} friction
        </span>
        <span class="badge" style="background:#FFFFFF11;color:#CCCCCC;border:1px solid #444">
          😤 {fm.get('emotional_signature', 'N/A')}
        </span>
        <span class="badge"
          style="background:#FFFFFF11;color:#CCCCCC;border:1px solid #444">
          📊 Impact Score: {impact_score}
        </span>
      </div>
      <div class="fm-section-label">Root Cause</div>
      <div class="fm-root-cause">{fm.get('root_cause', '')}</div>
      <div class="fm-section-label">PM Insight</div>
      <div class="fm-pm-insight">💡 {fm.get('pm_insight', '')}</div>
    {jtbd_html}
    </div>
    """, unsafe_allow_html=True)

    quotes = fm.get("representative_quotes", [])
    if quotes:
        with st.expander(f"📣 User voices ({len(quotes)} quotes)", expanded=False):
            for q in quotes[:3]:
                src_label = {"app_store": "App Store", "play_store": "Play Store",
                             "reddit": "Reddit", "reddit_comment": "Reddit",
                             "spotify_community": "Spotify Community",
                             "twitter": "Twitter/X", "twitter_apify": "Twitter/X",
                             "seed": "Synthetic seed (demo)",
                             }.get(q.get("source", ""), "Review")
                is_synthetic = q.get("provenance") == "synthetic_seed" or q.get("source") == "seed"
                synthetic_tag = ' <span style="color:#FFB800;font-weight:600">⚠ synthetic, not a real review</span>' if is_synthetic else ""
                rating_str = f"★ {q['rating']}" if q.get("rating") else ""
                st.markdown(f"""
                <div class="quote-block">
                  "{q['text']}"
                  <div class="quote-meta">{src_label}{synthetic_tag} {rating_str} · {q.get('helpful_votes', 0)} helpful votes</div>
                </div>
                """, unsafe_allow_html=True)


# ── Interview guide ────────────────────────────────────────────────────────────

def render_interview_guide(guide):
    if not guide:
        return
    st.markdown('<div class="section-header">User Interview Guide · JTBD Framework</div>', unsafe_allow_html=True)

    status = guide.get("synthesis_status", "live")
    if status != "live":
        banner_text = {
            "fallback_demo": "⚠️ FALLBACK CONTENT — live guide generation failed; this is pre-written placeholder content.",
            "demo_mode": "ℹ️ DEMO MODE — illustrative guide, not generated from a live pipeline run.",
        }.get(status, "⚠️ FALLBACK CONTENT — not generated live.")
        st.markdown(f'<div class="fallback-banner">{banner_text}</div>', unsafe_allow_html=True)

    # Segmentation honesty disclosure: the "Comfort Zone Listener" persona
    # is a TARGET for these interviews to validate, not something derived
    # from the review data itself — App Store / Play Store / Reddit
    # reviews carry no age, tenure, or usage-frequency signal, so this
    # screener is a recruiting hypothesis, not a confirmed segment.
    st.markdown("""
    <div style="font-size:12px;color:#888;line-height:1.6;margin-bottom:16px;padding:10px 14px;
         background:rgba(255,255,255,0.025);border-radius:8px;border-left:2px solid #555;">
      📋 The persona and screener criteria below are a <strong style="color:#bbb">target segment to validate</strong>
      in these interviews — review data has no age/tenure/usage-frequency fields,
      so this is a recruiting hypothesis, not something the review analysis itself confirmed.
    </div>
    """, unsafe_allow_html=True)

    objective = guide.get("interview_objective", "")
    if objective:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:10px;padding:18px 22px;margin-bottom:20px;">
          <div class="fm-section-label">Objective</div>
          <div style="font-size:14px;color:#E0E0E0;line-height:1.6">{objective}</div>
        </div>
        """, unsafe_allow_html=True)
    for q in guide.get("questions", []):
        target = q.get("failure_mode_targeted", "")
        st.markdown(f"""
        <div class="interview-card">
          <div class="fm-section-label">Q{q['number']} · Targets: {target}</div>
          <div class="interview-q">{q['question']}</div>
          {''.join([f'<div class="interview-probe">↳ {p}</div>' for p in q.get('follow_ups', [])])}
        </div>
        """, unsafe_allow_html=True)
    closing = guide.get("closing_question", "")
    if closing:
        st.markdown(f"""
        <div style="background:rgba(29,185,84,0.06);border:1px solid rgba(29,185,84,0.3);border-radius:10px;padding:18px 22px;margin-top:8px;">
          <div class="fm-section-label" style="color:{SPOTIFY_GREEN}">Closing Question</div>
          <div style="font-size:14px;font-weight:500;color:{TEXT_PRIMARY};line-height:1.5">{closing}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(f"""<div style="font-family:'Space Grotesk',sans-serif;font-size:13px;font-weight:700;
             letter-spacing:1.5px;text-transform:uppercase;color:{SPOTIFY_GREEN};margin-bottom:16px">
          Engine Controls</div>""", unsafe_allow_html=True)

        st.markdown("**Data Sources**")
        # App Store, Play Store, and Community need no credentials and are
        # keyword-filtered for discovery relevance — defaulted ON so a
        # first run produces real results out of the box. Reddit/Twitter
        # need an Apify export or bearer token, so they default OFF.
        use_appstore  = st.toggle("App Store (iOS)", value=True)
        use_play      = st.toggle("Google Play Store", value=True)
        use_community = st.toggle("Spotify Community Forums", value=True)
        use_reddit    = st.toggle("Reddit — Apify JSON", value=False)
        use_twitter_apify = st.toggle("Twitter/X — Apify JSON", value=False)
        use_twitter   = st.toggle("Twitter/X — Live API", value=False)

        st.divider()
        use_seed = st.toggle("Synthetic Seed Reviews (25, offline demo)", value=False)
        if use_seed:
            st.caption("⚠️ Synthetic placeholder content, not real reviews. Tagged `provenance=synthetic_seed` "
                       "in the data and visibly flagged on any card it touches — use only when no live "
                       "source is reachable, never blend into a 'reviews analyzed' headline without disclosure.")

        apify_reddit_path = None
        if use_reddit:
            apify_reddit_path = st.text_input("Reddit Apify JSON file path", value="data/apify_reddit.json")
            st.caption("No Reddit API needed — uses your Apify export")

        apify_twitter_path = None
        if use_twitter_apify:
            apify_twitter_path = st.text_input("Twitter Apify JSON file path", value="data/apify_twitter.json")
            st.caption("No Twitter API needed — uses your Apify export")

        if use_twitter:
            st.caption("Requires TWITTER_BEARER_TOKEN in .env — free tier returns 402 (paywalled), will likely fail")
        if use_appstore or use_play or use_community:
            st.caption("Live scrapers may take 30-90s")

        st.divider()
        st.markdown("**Discovery Relevance Filter**")
        filter_discovery = st.toggle("Keyword-filter App/Play Store for discovery relevance", value=True)
        st.caption("Keep ON — without this, App/Play Store reviews are dominated by ads/bugs/pricing "
                   "complaints (the most common topic in any freemium app's reviews), drowning out "
                   "the discovery signal this project measures.")

st.divider()
        st.markdown("**API Key**")
        
        # SECURITY FIX: If the key is in Streamlit secrets, load it silently in the backend.
        # NEVER pass a secret to the value= parameter of a UI text box.
        if "GROQ_API_KEY" in st.secrets:
            st.success("✅ API securely connected.")
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        else:
            # Fallback for local testing if secrets aren't configured
            api_key_input = st.text_input(
                "Groq API Key", type="password",
                value=os.getenv("GROQ_API_KEY", ""),
                placeholder="gsk_...",
                help="Get a free key at console.groq.com"
            )
            if api_key_input:
                os.environ["GROQ_API_KEY"] = api_key_input

        st.divider()
        run_btn = st.button("▶  Run Engine", use_container_width=True)
        st.divider()
        st.markdown(f"""<div style="font-size:11px;color:#888;line-height:1.6">
          PM Fellowship Project<br>Spotify Growth Team<br>
          North Star: Weekly Active Discovery Rate<br><br>
          <span style="color:#666">Anonymous · No PII stored</span></div>""", unsafe_allow_html=True)

    return (run_btn, use_seed, use_appstore, use_play, use_reddit, use_community,
            use_twitter, use_twitter_apify, filter_discovery,
            apify_reddit_path, apify_twitter_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    (run_btn, use_seed, use_appstore, use_play, use_reddit, use_community,
     use_twitter, use_twitter_apify, filter_discovery,
     apify_reddit_path, apify_twitter_path) = render_sidebar()

    # Pipeline trigger — runs before anything else is rendered so a fresh
    # result is available immediately on rerun.
    if run_btn:
        with st.spinner(""):
            result = run_full_pipeline(
                use_seed=use_seed, use_appstore=use_appstore,
                use_playstore=use_play, use_reddit=use_reddit,
                use_community=use_community, use_twitter=use_twitter,
                use_twitter_apify=use_twitter_apify,
                filter_discovery=filter_discovery,
                apify_reddit_path=apify_reddit_path,
                apify_twitter_path=apify_twitter_path,
            )
        if result is not None:
            st.session_state["failure_modes_data"] = result
            st.success("✓ Engine complete.")
            time.sleep(0.5)
            st.rerun()
        # If result is None (no reviews collected), run_full_pipeline already
        # rendered a clear error — fall through and show whatever existing
        # data is available rather than crashing.

    # Load data — BEFORE the hero renders, so the hero's stat pills can
    # show this run's real numbers instead of a hardcoded claim. The
    # original hero always said "1,700+ reviews analyzed" and "5 failure
    # modes identified" regardless of what (if anything) had actually run.
    data = st.session_state.get("failure_modes_data") or load_failure_modes()
    if data is None:
        from synthesizer import DEMO_FAILURE_MODES
        os.makedirs("data", exist_ok=True)
        with open("data/failure_modes.json", "w") as f:
            json.dump(DEMO_FAILURE_MODES, f)
        data = DEMO_FAILURE_MODES
        st.session_state["failure_modes_data"] = data

    failure_modes = data.get("failure_modes", [])
    failure_modes = sorted(failure_modes, key=lambda x: x.get("impact_score", 0), reverse=True)
    master          = data.get("master_synthesis", {})
    interview_guide = data.get("interview_guide", {})
    meta            = data.get("metadata", {})
    total_reviews   = meta.get("total_reviews_analyzed", sum(fm.get("review_count", 0) for fm in failure_modes))
    n_clusters      = len(failure_modes)
    n_seed_disclosed    = meta.get("n_reviews_synthetic_seed")
    n_scraped_disclosed = meta.get("n_reviews_scraped")
    is_demo_dataset = meta.get("synthesis_status") == "demo_mode" or all(
        fm.get("synthesis_status") == "demo_mode" for fm in failure_modes
    ) if failure_modes else False

    # Hero — stat pills now reflect this run's actual data, not a fixed claim.
    review_pill = f"{total_reviews:,} reviews analyzed"
    if n_seed_disclosed is not None and n_scraped_disclosed is not None and n_seed_disclosed > 0:
        review_pill = f"{total_reviews:,} reviews ({n_scraped_disclosed:,} scraped + {n_seed_disclosed:,} synthetic)"
    cluster_pill = f"{n_clusters} failure mode{'s' if n_clusters != 1 else ''} identified (data-driven count)"
    demo_pill = '<span class="stat-pill" style="background:rgba(255,184,0,0.12);border-color:rgba(255,184,0,0.4);color:#FFB800">⚠ Demo dataset — not a live run</span>' if is_demo_dataset else ""

    st.markdown(f"""
    <div class="hero">
      <div class="hero-eyebrow">Spotify · PM Fellowship · Review Intelligence Engine</div>
      <div class="hero-title">The Discovery Paradox, Made Legible</div>
      <div class="hero-subtitle">
        This engine ingests App Store, Play Store, Reddit (Apify), Spotify Community Forums,
        and Twitter/X signals — keyword-filtered for discovery relevance — embeds them
        semantically, clusters them into a data-driven number of failure modes, and uses
        Groq to name, explain, and prioritize each one, converting unstructured user
        frustration into actionable PM insight.
      </div>
      <div class="stat-row">
        <span class="stat-pill">{review_pill}</span>
        <span class="stat-pill">{cluster_pill}</span>
        <span class="stat-pill">JTBD interview guide generated</span>
        <span class="stat-pill">LLM: Groq</span>
        {demo_pill}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Master synthesis box
    if master:
        m_status = master.get("synthesis_status", "live")
        m_box_class = "synthesis-box synthesis-box-flagged" if m_status != "live" else "synthesis-box"
        m_banner = ""
        if m_status != "live":
            m_banner_text = {
                "fallback_demo": "⚠️ FALLBACK CONTENT — live master synthesis failed; this narrative is pre-written placeholder content, not generated from this run's failure modes.",
                "demo_mode": "ℹ️ DEMO MODE — illustrative synthesis, not generated from a live pipeline run.",
            }.get(m_status, "⚠️ FALLBACK CONTENT — not generated live.")
            m_banner = f'<div class="fallback-banner" style="margin-bottom:20px">{m_banner_text}</div>'

        st.markdown(f"""
        <div class="{m_box_class}">
          {m_banner}
          <div class="synthesis-title">"{master.get('master_insight_title', '')}"</div>
          <div class="synthesis-text">{master.get('connecting_narrative', '')}</div>
          <div class="synthesis-label">Strategic Implication</div>
          <div class="synthesis-text" style="margin-bottom:14px">{master.get('strategic_implication', '')}</div>
          <div class="synthesis-label">Highest-Leverage Intervention</div>
          <div class="synthesis-text" style="margin-bottom:0">
            <strong style="color:{SPOTIFY_GREEN}">{master.get('highest_leverage_mode', '')}</strong>
            — {master.get('highest_leverage_rationale', '')}
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Charts row
    if failure_modes:
        col1, col2 = st.columns([1.4, 1])
        with col1:
            st.markdown('<div class="section-header">Review Volume by Failure Mode</div>', unsafe_allow_html=True)
            render_impact_chart(failure_modes)
        with col2:
            st.markdown('<div class="section-header">Coverage Summary</div>', unsafe_allow_html=True)
            # Total reviews banner — discloses scraped vs. synthetic seed
            # provenance when known, instead of presenting one opaque total.
            provenance_line = ""
            if n_seed_disclosed is not None and n_scraped_disclosed is not None:
                if n_seed_disclosed > 0:
                    provenance_line = f'<div style="font-size:11px;color:#FFB800;margin-top:4px">⚠ includes {n_seed_disclosed} synthetic seed reviews</div>'
                else:
                    provenance_line = f'<div style="font-size:11px;color:#666;margin-top:4px">100% scraped, 0 synthetic</div>'
            st.markdown(f"""
            <div style="background:{CARD_BG};border:1px solid {CARD_BORDER};border-radius:10px;
                 padding:14px 20px;margin-bottom:12px;display:flex;align-items:center;gap:16px">
              <div>
                <div style="font-family:Space Grotesk,sans-serif;font-size:32px;font-weight:700;
                     color:{SPOTIFY_GREEN};line-height:1">{total_reviews:,}</div>
                <div style="font-size:12px;color:#BBBBBB;margin-top:3px">total reviews analyzed</div>
                {provenance_line}
              </div>
              <div style="border-left:1px solid #3A3A3A;padding-left:16px">
                <div style="font-size:12px;color:#BBBBBB">Sources (varies by run)</div>
                <div style="font-size:13px;color:#E0E0E0;margin-top:2px">App Store · Play Store · Reddit · Community · Twitter</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
            # Per-cluster tiles — however many clusters this run produced
            n_colors = len(CLUSTER_ACCENT_COLORS)
            cols_inner = st.columns(2)
            for i, fm in enumerate(failure_modes):
                with cols_inner[i % 2]:
                    impact = fm.get("business_impact", "MEDIUM")
                    impact_color = IMPACT_COLORS.get(impact, "#888")
                    flagged_tag = ' <span style="color:#FFB800">⚠</span>' if fm.get("synthesis_status", "live") != "live" else ""
                    st.markdown(f"""
                    <div class="coverage-card" style="margin-bottom:10px">
                      <div class="coverage-pct" style="color:{CLUSTER_ACCENT_COLORS[i % n_colors]}">{fm.get('pct_of_total','?')}%</div>
                      <div class="coverage-name">{fm.get('failure_mode_name','')}{flagged_tag}</div>
                      <div class="coverage-meta">
                        <span style="color:{impact_color}">{impact}</span> impact ·
                        {fm.get('review_count', '?')} reviews
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Scatter plot
        df_clustered = load_clustered_reviews()
        if df_clustered is not None and "umap_x" in df_clustered.columns and len(df_clustered) > 0:
            st.markdown('<div class="section-header">Semantic Review Cluster Map</div>', unsafe_allow_html=True)
            st.markdown("<p style='font-size:13px;color:#AAAAAA;margin-bottom:8px'>Each dot is one review. Clusters show semantic similarity — reviews expressing the same underlying frustration group together.</p>", unsafe_allow_html=True)
            render_cluster_scatter(df_clustered, failure_modes)
        else:
            st.markdown('<div class="section-header">Semantic Review Cluster Map</div>', unsafe_allow_html=True)
            st.markdown(f"<p style='font-size:13px;color:#888'>Run the pipeline to generate the cluster map.</p>", unsafe_allow_html=True)

    # Failure mode cards
    n = len(failure_modes)
    st.markdown(f'<div class="section-header">The {n} Discovery Failure Mode{"s" if n != 1 else ""} (Data-Driven Count)</div>', unsafe_allow_html=True)
    for i, fm in enumerate(failure_modes):
        render_failure_mode_card(fm, i)

    # Interview guide
    if interview_guide:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        render_interview_guide(interview_guide)

    # Footer
    st.markdown(f"""
    <div style="margin-top:48px;padding-top:24px;border-top:1px solid {CARD_BORDER};
         text-align:center;font-size:12px;color:#888;letter-spacing:0.5px">
      Spotify PM Fellowship · Review Intelligence Engine ·
      North Star: Weekly Active Discovery Rate (new artists saved ÷ total sessions/week) ·
      Anonymous — no PII stored
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
