"""
synthesizer.py — LLM Synthesis Layer (Groq Edition)
Sends cluster quotes to Groq API and extracts Discovery Failure Modes
with names, root causes, emotional signatures, and PM insights. The
number of failure modes is whatever clusterer.py's natural clustering
produced — not a fixed count.

Every output object (per-cluster failure mode, master synthesis,
interview guide) carries a `synthesis_status` field: "live" if it came
from a successful LLM call, or "fallback_demo" / "fallback_generic" if
live synthesis failed and pre-written content was substituted. This
exists so a failed synthesis can never be displayed identically to a
genuine one — see run_synthesis_pipeline() and the matching UI handling
in app.py.
"""

import os
import json
import time
import logging
import pandas as pd  
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────

CLUSTER_NAMING_PROMPT = """You are a senior Product Manager at Spotify analyzing user review clusters for a discovery strategy project.

I will give you a cluster of user reviews (grouped by semantic similarity). Your job is to:
1. Give this cluster a sharp, memorable "failure mode" name (3-6 words, noun phrase)
2. Write a one-sentence root cause (what's actually broken at the system level)
3. Identify the dominant emotional signature (1-2 words: e.g. "frustrated resignation", "nostalgic longing", "anxious overwhelm")
4. Rate the business impact: HIGH / MEDIUM / LOW (on churn and LTV)
5. Write one PM insight: what feature or intervention would fix this
6. Write a 2-sentence user quote synthesis (capturing the essence without quoting directly)
7. Extract the underlying Job-to-Be-Done these users are retrospectively trying to accomplish — phrase it in the JTBD format: "When [situation], I want to [motivation], so I can [expected outcome]." Base this on what the reviews actually describe wanting, not a generic discovery JTBD.

CLUSTER REVIEWS:
{quotes}

Respond ONLY in this exact JSON format (no preamble, no markdown, no code fences):
CRITICAL RULE: Output strictly valid JSON. Escape all internal quotes using a backslash (\").{{
  "failure_mode_name": "...",
  "tagline": "one sharp sentence describing this failure mode",
  "root_cause": "...",
  "emotional_signature": "...",
  "business_impact": "HIGH|MEDIUM|LOW",
  "business_impact_rationale": "...",
  "pm_insight": "...",
  "user_voice_synthesis": "...",
  "jobs_to_be_done": "When ..., I want to ..., so I can ...",
  "discovery_friction_type": "algorithmic|ux|social|contextual|cognitive"
}}"""


MASTER_SYNTHESIS_PROMPT = """You are the lead PM researcher at Spotify. You've just analyzed {total_reviews} user reviews clustered into {n_clusters} semantic groups. Each group has been labeled with a failure mode.

Here are the {n_clusters} failure modes with their details:
{failure_modes_json}

Your task is to write the executive synthesis for a PM fellowship deck slide. Produce:
1. An overarching insight title (assertion-driven, max 12 words)
2. A 3-sentence narrative connecting all {n_clusters} failure modes into one systems-level story
3. The #1 strategic implication for Spotify's Growth Team
4. Which failure mode is the highest-leverage intervention point and why

Respond ONLY in JSON format (no preamble, no markdown, no code fences):
{{
  "master_insight_title": "...",
  "connecting_narrative": "...",
  "strategic_implication": "...",
  "highest_leverage_mode": "failure_mode_name",
  "highest_leverage_rationale": "..."
}}"""


INTERVIEW_GUIDE_PROMPT = """Based on these {n_clusters} discovery failure modes from Spotify user research:

{failure_modes_summary}

Write a user interview guide for the "Comfort Zone Listener" persona — the TARGET segment proposed for upcoming validation interviews (Spotify Premium users, age 25-35, urban, 2+ years on platform, high-tenure but low discovery rate). Note: this persona is a hypothesis to validate, not something the review data itself confirms — App Store, Play Store, and Reddit reviews carry no age/tenure/usage-frequency signal, so frame screener criteria as what to check for in recruiting, not as an established fact.

The guide should:
- Be {n_questions} questions long
- Follow Jobs-to-Be-Done methodology
- Each question probes a different failure mode
- Include follow-up probes for each question
- Avoid leading questions
- Focus on behavior, not opinions

Respond ONLY in JSON format (no preamble, no markdown, no code fences):
CRITICAL RULE: Output strictly valid JSON. Escape all internal quotes using a backslash (\").{{
  "interview_objective": "...",
  "screener_criteria": ["...", "..."],
  "questions": [
    {{
      "number": 1,
      "question": "...",
      "failure_mode_targeted": "...",
      "follow_ups": ["...", "..."]
    }}
  ],
  "closing_question": "..."
}}"""


# ─────────────────────────────────────────────
# Groq API Call
# ─────────────────────────────────────────────

# Groq model
LLM_MODEL = "llama-3.3-70b-versatile"

def call_llm(prompt: str, max_tokens: int = 1000, retry_count: int = 0) -> str:
    """Call Groq API and return text response."""
    import os
    import time
    from groq import Groq
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in environment (.env file)")

    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=LLM_MODEL, # Meta's massive 70B model, running incredibly fast
            response_format={"type": "json_object"},
           messages=[
    {
        "role": "system",
        "content": """
You are a senior Product Manager.


Rules:
- No markdown
- No code fences
- No explanations
- No comments
- No trailing commas
- Escape all quotes
- Escape all backslashes
- Must be accepted by Python json.loads()
"""
    },
    {
        "role": "user",
        "content": prompt
    }
],
            max_tokens=max_tokens,
            temperature=0.3,
                    )
        return response.choices[0].message.content
        
    except Exception as e:
        err_str = str(e).lower()
        # Groq's free tier has a Requests Per Minute limit.
        if retry_count < 3 and ("429" in err_str or "rate limit" in err_str):
            import logging
            log = logging.getLogger(__name__)
            log.warning(f"Groq rate limit hit. Waiting 10 seconds to retry {retry_count + 1}/3...")
            time.sleep(10) # Groq limits reset quickly, so we only need a short pause
            return call_llm(prompt, max_tokens=max_tokens, retry_count=retry_count + 1)
        raise
    
def parse_json_response(text: str) -> dict:
    """
    Parse JSON from Groq response with aggressive repair.

    Handles:
    - markdown fences
    - extra text around JSON
    - smart quotes
    - invalid escape sequences
    - truncated JSON
    """
    import json
    import re

    text = text.strip()

    # Remove markdown fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text)
        text = text.strip()

    # Extract JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)

    # Replace smart quotes
    text = (
        text.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
            .replace("\u2019", "'")
    )

    try:
        return json.loads(text)

    except json.JSONDecodeError as e:

        log.warning(
            f"Initial JSON parse failed ({e}) — attempting repair..."
        )

        # Fix invalid escape sequences
        text = re.sub(
            r'\\(?!["\\/bfnrtu])',
            r'\\\\',
            text
        )

        # Repair truncation
        text = _repair_truncated_json(text)

        try:
            return json.loads(text)

        except Exception as e2:
            log.error("RAW LLM RESPONSE:")
            log.error(text)
            log.error(f"Repair failed: {e2}")
            raise


def _repair_truncated_json(text: str) -> str:
    """
    Best-effort repair for JSON truncated mid-string or mid-object because
    the model hit its token limit. Closes any unterminated string and pads
    missing closing brackets/braces so json.loads() can succeed.
    """
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string

    if in_string:
        text = text.rstrip()
        if text.endswith("\\"):
            text = text[:-1]
        text += '"'

    text = text.rstrip()
    if text.endswith(","):
        text = text[:-1]

    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    text += "]" * max(open_brackets, 0)
    text += "}" * max(open_braces, 0)

    return text


# ─────────────────────────────────────────────
# Per-Cluster Synthesis
# ─────────────────────────────────────────────

def synthesize_cluster(cluster_id: int, cluster_meta: dict) -> dict:
    """Send one cluster's quotes to Groq and get failure mode label."""
    quotes = cluster_meta["representative_quotes"]
    quotes_text = "\n\n".join([
        f'Review {i+1} (source: {q["source"]}, rating: {q.get("rating", "N/A")}):\n"{q["text"]}"'
        for i, q in enumerate(quotes[:5])
    ])

    prompt = CLUSTER_NAMING_PROMPT.format(quotes=quotes_text)
    log.info(f"  Synthesizing cluster {cluster_id} ({cluster_meta['size']} reviews) via Groq...")

    raw = call_llm(prompt, max_tokens=1000)
    synthesis = parse_json_response(raw)
    synthesis["cluster_id"] = cluster_id
    synthesis["review_count"] = cluster_meta["size"]
    synthesis["pct_of_total"] = cluster_meta["pct_of_total"]
    synthesis["avg_rating"] = cluster_meta.get("avg_rating")
    quotes = cluster_meta.get("representative_quotes", [])
    total_helpful = sum(
        q.get("helpful_votes", 0)
        for q in quotes
)   
    synthesis["total_helpful_votes"] = total_helpful
    synthesis["representative_quotes"] = cluster_meta["representative_quotes"][:3]

    return synthesis


# ─────────────────────────────────────────────
# Master Synthesis
# ─────────────────────────────────────────────

def synthesize_all_failure_modes(failure_modes: list[dict], total_reviews: int = None) -> dict:
    """Generate the master narrative connecting all failure modes."""
    summary = json.dumps([{
        "failure_mode_name": fm["failure_mode_name"],
        "root_cause": fm["root_cause"],
        "emotional_signature": fm["emotional_signature"],
        "business_impact": fm["business_impact"],
        "pct_of_reviews": fm.get("pct_of_total"),
    } for fm in failure_modes], indent=2)

    n_clusters = len(failure_modes)
    if total_reviews is None:
        total_reviews = sum(fm.get("review_count", 0) for fm in failure_modes)

    prompt = MASTER_SYNTHESIS_PROMPT.format(
        failure_modes_json=summary,
        n_clusters=n_clusters,
        total_reviews=total_reviews,
    )
    log.info("Synthesizing master narrative via Groq...")
    raw = call_llm(prompt, max_tokens=1000)
    return parse_json_response(raw)


# ─────────────────────────────────────────────
# Interview Guide Generation
# ─────────────────────────────────────────────

def generate_interview_guide(failure_modes: list[dict]) -> dict:
    """Generate a JTBD interview guide from the failure modes."""
    summary = "\n".join([
        f"{i+1}. {fm['failure_mode_name']}: {fm['root_cause']}"
        for i, fm in enumerate(failure_modes)
    ])
    n_clusters = len(failure_modes)
    # Clamp question count to a practical interview length regardless of
    # how many clusters the data produced (a 12-question guide from 12
    # clusters would be unusable in a real interview slot).
    n_questions = max(5,n_clusters)

    prompt = INTERVIEW_GUIDE_PROMPT.format(
        failure_modes_summary=summary,
        n_clusters=n_clusters,
        n_questions=n_questions,
    )
    log.info("Generating interview guide Groq...")
    raw = call_llm(prompt, max_tokens=1000)
    return parse_json_response(raw)


# ─────────────────────────────────────────────
# Full Synthesis Pipeline
# ─────────────────────────────────────────────

def run_synthesis_pipeline(
    clusters_meta: dict,
    output_path: str = "data/failure_modes.json",
    generate_guide: bool = True,
    retry_cooldown_seconds: int = 8,
) -> dict:
    """
    Orchestrates the synthesis layer.

    Every failure mode, the master synthesis, and the interview guide are
    tagged with a `synthesis_status` field:
      "live"            — came from a successful Groq call
      "fallback_demo"   — live synthesis failed (even after one retry);
                           pre-written placeholder content was substituted
      "fallback_generic" — live synthesis failed and no matching demo
                           content existed either; a minimal generic
                           placeholder was used

    This exists because the previous version silently merged fabricated
    DEMO_FAILURE_MODES content with real review counts and displayed it
    identically to genuine synthesis — a failure mode card could show a
    real review count next to a description with zero relationship to
    the actual reviews in that cluster, with no flag anywhere. app.py's
    rendering functions check this field and visibly mark fallback
    content; they must never present it as equivalent to live synthesis.
    """
    import json
    from pathlib import Path

    failure_modes = []

    for cluster_id, meta in clusters_meta.items():
        live_fm = None
        last_error = None

        # Attempt 1
        try:
            log.info(f"Synthesizing cluster {cluster_id} via live Groq...")
            live_fm = synthesize_cluster(int(cluster_id), meta)
        except Exception as e:
            last_error = e
            log.error(f"Live synthesis failed for cluster {cluster_id}: {e}")

        # Attempt 2 — one retry after a short cooldown. Groq's free-tier
        # rate limit is the most common transient failure here, and it
        # often clears within a few seconds.
        if live_fm is None:
            log.warning(f"Retrying cluster {cluster_id} once after a {retry_cooldown_seconds}s cooldown...")
            time.sleep(retry_cooldown_seconds)
            try:
                live_fm = synthesize_cluster(int(cluster_id), meta)
            except Exception as e:
                last_error = e
                log.error(f"Retry also failed for cluster {cluster_id}: {e}")

        if live_fm is not None:
            if "business_impact" not in live_fm:
                live_fm["business_impact"] = "MEDIUM"
            live_fm["synthesis_status"] = "live"
            failure_modes.append(live_fm)
            continue

        # Both attempts failed — fall back, but flag it loudly.
        log.warning(
            f"⚠️  Cluster {cluster_id} synthesis FAILED after retry ({last_error}). "
            f"Substituting placeholder content — this MUST be flagged in the UI, "
            f"never displayed as genuine synthesis."
        )

        demo_match = [m for m in DEMO_FAILURE_MODES.get("failure_modes", []) if m.get("cluster_id") == int(cluster_id)]
        if demo_match:
            # Copy — never mutate the shared module-level DEMO_FAILURE_MODES
            # dict in place. The original code wrote directly into
            # demo_match[0], which silently corrupted the module-level
            # constant across repeated runs within the same process.
            match = dict(demo_match[0])
            match["review_count"] = meta.get("size", match.get("review_count", 0))
            match["pct_of_total"] = meta.get("pct_of_total", match.get("pct_of_total", 0.0))
            match["synthesis_status"] = "fallback_demo"
            match["fallback_reason"] = str(last_error)
            failure_modes.append(match)
        else:
            failure_modes.append({
                "cluster_id": int(cluster_id),
                "failure_mode_name": f"Synthesis Pending — Cluster {int(cluster_id) + 1}",
                "tagline": "Live synthesis failed and no fallback content exists for this cluster.",
                "root_cause": "Not yet generated. Re-run the engine to attempt synthesis again.",
                "emotional_signature": "unknown",
                "business_impact": "MEDIUM",
                "review_count": meta.get("size", 0),
                "pct_of_total": meta.get("pct_of_total", 0.0),
                "representative_quotes": meta.get("representative_quotes", [])[:3],
                "synthesis_status": "fallback_generic",
                "fallback_reason": str(last_error),
            })

    total_reviews = sum(meta.get("size", 0) for meta in clusters_meta.values())

    try:
        master = synthesize_all_failure_modes(failure_modes, total_reviews=total_reviews)
        master["synthesis_status"] = "live"
    except Exception as e:
        log.error(f"Master narrative synthesis failed: {e}. Falling back to pre-computed narrative.")
        master = dict(DEMO_FAILURE_MODES.get("master_synthesis", {}))
        master["synthesis_status"] = "fallback_demo"
        master["fallback_reason"] = str(e)

    interview_guide = {}
    if generate_guide:
        try:
            interview_guide = generate_interview_guide(failure_modes)
            interview_guide["synthesis_status"] = "live"
        except Exception as e:
            log.error(f"Guide synthesis failed: {e}. Falling back.")
            interview_guide = dict(DEMO_FAILURE_MODES.get("interview_guide", {}))
            interview_guide["synthesis_status"] = "fallback_demo"
            interview_guide["fallback_reason"] = str(e)

    # -----------------------------
    # Quantitative Impact Scoring
    # -----------------------------

    max_reviews = max((fm.get("review_count", 0) for fm in failure_modes), default=0) or 1
    max_helpful = max((fm.get("total_helpful_votes", 0) for fm in failure_modes), default=0)

    for fm in failure_modes:

        volume_score = (fm.get("review_count", 0) / max_reviews) * 100

        avg_rating = fm.get("avg_rating") or 3
        severity_score = ((5 - avg_rating) / 4) * 100

        helpful_score = 0
        if max_helpful > 0:
            helpful_score = (fm.get("total_helpful_votes", 0) / max_helpful) * 100

        impact_score = (
            volume_score * 0.50
            + severity_score * 0.30
            + helpful_score * 0.20
        )

        fm["impact_score"] = round(impact_score, 1)

        if impact_score >= 70:
            fm["business_impact"] = "HIGH"
        elif impact_score >= 40:
            fm["business_impact"] = "MEDIUM"
        else:
            fm["business_impact"] = "LOW"

    n_live = sum(1 for fm in failure_modes if fm.get("synthesis_status") == "live")
    n_fallback = len(failure_modes) - n_live

    output = {
        "master_synthesis": master,
        "failure_modes": failure_modes,
        "interview_guide": interview_guide,
        "metadata": {
            "total_reviews_analyzed": total_reviews,
            "n_clusters": len(clusters_meta),
            "n_clusters_live_synthesis": n_live,
            "n_clusters_fallback_synthesis": n_fallback,
            "llm_provider": "Groq",
        }
    }

    if n_fallback > 0:
        log.warning(
            f"⚠️  {n_fallback}/{len(failure_modes)} failure modes used fallback content. "
            f"Check synthesis_status on each card before presenting this run."
        )

    # Save clean, valid output immediately
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    return output

# ─────────────────────────────────────────────
# Hardcoded Failure Modes (Demo / Fallback)
# ─────────────────────────────────────────────
# Used when no API key is set. Derived from the seed review analysis.

DEMO_FAILURE_MODES = {
    "failure_modes": [
        {
            "cluster_id": 0,
            "failure_mode_name": "Algorithm Comfort Lock",
            "tagline": "The recommendation engine learned your taste so well it stopped challenging it.",
            "root_cause": "Collaborative filtering optimizes for immediate engagement signals (plays, saves) rather than long-term discovery velocity, creating a self-reinforcing loop that narrows the user's sonic profile over time.",
            "emotional_signature": "nostalgic frustration",
            "business_impact": "HIGH",
            "business_impact_rationale": "Users aware of this paradox are most likely to churn to competitors or seek discovery elsewhere, reducing LTV and brand differentiation.",
            "pm_insight": "Introduce a 'Discovery Mode' toggle that deliberately deprioritizes familiar artists and injects controlled novelty — with skip-forgiveness so exploration isn't punished.",
            "user_voice_synthesis": "Users remember a time when the algorithm felt adventurous, and mourn its regression into a comfortable mirror of their past taste. They describe the current experience as sophisticated stagnation.",
            "discovery_friction_type": "algorithmic",
            "review_count": 549,
            "pct_of_total": 32.0,
            "representative_quotes": [
                {"text": "I went from discovering 10 new artists a month to basically zero. The algorithm learned what I like then stopped trying.", "source": "seed", "rating": 2},
                {"text": "Discover Weekly was almost magical in 2019. Something changed. The playlists are safer now. I miss the algorithm that took risks.", "source": "seed", "rating": 3},
                {"text": "I made a deliberate effort to listen to unfamiliar genres for a month. After 30 days, Discover Weekly gave me... my usual stuff.", "source": "seed", "rating": 2},
            ]
        },
        {
            "cluster_id": 1,
            "failure_mode_name": "Passive Listening Drift",
            "tagline": "Autoplay makes discovery effortless to avoid — and almost impossible to pursue.",
            "root_cause": "The app's zero-friction passive listening mode creates a default state of non-discovery; autoplay transitions are optimized for session continuation, not musical growth.",
            "emotional_signature": "guilty detachment",
            "business_impact": "HIGH",
            "business_impact_rationale": "Passive listeners have high session time but low discovery KPIs; they're at risk of not feeling Spotify delivers value beyond a cheaper background noise service.",
            "pm_insight": "Design intentional 'Discovery Checkpoints' — moments where autoplay pauses and surfaces a genuinely unfamiliar track with a clear accept/skip choice, framed as a game rather than friction.",
            "user_voice_synthesis": "Users acknowledge passively surrendering agency to the algorithm, describing the realization — often via Wrapped — that an entire year passed with almost no new discoveries despite thousands of listening hours.",
            "discovery_friction_type": "ux",
            "review_count": 412,
            "pct_of_total": 24.0,
            "representative_quotes": [
                {"text": "My Wrapped showed I listened to 40,000 minutes and discovered 3 new artists. Three. In a year.", "source": "seed", "rating": 3},
                {"text": "Autoplay kicks in and suddenly I'm three hours deep in generic acoustic chill music I never chose.", "source": "seed", "rating": 2},
                {"text": "The app makes it so easy to just let music play that I never actively choose anything anymore.", "source": "seed", "rating": 3},
            ]
        },
        {
            "cluster_id": 2,
            "failure_mode_name": "Context-Blind Recommendations",
            "tagline": "Spotify knows your history but not your present moment.",
            "root_cause": "The recommendation system uses historical listening data as its primary signal, with no awareness of real-time context (mood, activity, environment, social setting) that should govern what music fits right now.",
            "emotional_signature": "contextual frustration",
            "business_impact": "MEDIUM",
            "business_impact_rationale": "Context mismatch reduces recommendation relevance and teaches users to distrust the algorithm, increasing manual playlist selection and reducing Spotify's differentiation from a dumb music library.",
            "pm_insight": "Build a lightweight context layer: a simple mood/activity selector at session start that temporarily overrides historical weight in the recommendation engine.",
            "user_voice_synthesis": "Users draw a clear distinction between who they were yesterday and who they need to be right now — and are frustrated that the algorithm only knows the former. They want music for the moment, not a playlist of their past selves.",
            "discovery_friction_type": "contextual",
            "review_count": 343,
            "pct_of_total": 20.0,
            "representative_quotes": [
                {"text": "I listen to focus music at work, party playlists on weekends. Spotify has no idea what I need right now — it just plays based on what I listened to yesterday.", "source": "seed", "rating": 3},
                {"text": "There's no way to say 'I want something I've never heard, in a direction I haven't gone, show me something weird and wonderful'.", "source": "seed", "rating": 2},
                {"text": "Perfect for background music when I don't care. Completely useless when I actually want to find something new.", "source": "seed", "rating": 4},
            ]
        },
        {
            "cluster_id": 3,
            "failure_mode_name": "Profile Contamination Spiral",
            "tagline": "One contextual listen or shared account can corrupt months of algorithmic learning.",
            "root_cause": "The algorithm treats all listening signals equally regardless of intent or context, making the taste profile fragile and non-recoverable when anomalous data enters — through shared accounts, parties, or situational listening.",
            "emotional_signature": "helpless betrayal",
            "business_impact": "MEDIUM",
            "business_impact_rationale": "Profile contamination events cause direct algorithm feature abandonment; affected users stop trusting Discover Weekly and Daily Mix, reducing engagement with Spotify's highest-differentiation features.",
            "pm_insight": "Introduce session tagging ('This wasn't me' / 'Ignore this session') and listener profile separation for households — protecting the algorithm's signal quality from accidental corruption.",
            "user_voice_synthesis": "Users describe algorithmic profile corruption as a violation — one unintentional listening event that permanently rewires months of carefully built recommendations, with no mechanism for correction or recovery.",
            "discovery_friction_type": "algorithmic",
            "review_count": 275,
            "pct_of_total": 16.0,
            "representative_quotes": [
                {"text": "I played a podcast for someone else in the car. Now my entire algorithmic profile is contaminated. Daily Mix started including true crime podcasts.", "source": "seed", "rating": 2},
                {"text": "My partner and I share an account. The algorithm is now a confused mess. We literally can't use any algorithmic features anymore.", "source": "seed", "rating": 3},
                {"text": "One playlist wrecked six months of algorithmic learning. Gym music invaded everything.", "source": "seed", "rating": 2},
            ]
        },
        {
            "cluster_id": 4,
            "failure_mode_name": "Discovery Entry Point Void",
            "tagline": "80 million songs and no on-ramp for the user who wants to explore something genuinely new.",
            "root_cause": "Spotify's information architecture assumes the user either knows what they want or is content with algorithmic curation — there is no designed experience for the intentional explorer who wants to leave their comfort zone but doesn't know where to start.",
            "emotional_signature": "paralyzed curiosity",
            "business_impact": "MEDIUM",
            "business_impact_rationale": "Users with unmet exploration intent represent a high-value opportunity; they are motivated to discover but unable to, making them prime candidates for a guided AI discovery feature.",
            "pm_insight": "Design an 'Explore Mode' with a conversational AI interface that asks 'What's the last song that made you feel something you haven't felt before?' and uses the answer to navigate the user to genuinely unfamiliar territory.",
            "user_voice_synthesis": "Users describe standing at the entrance of an infinite library with no map, librarian, or even a sense of which aisle to walk into. The paradox of choice collapses their motivation to explore, driving them back to the familiar.",
            "discovery_friction_type": "cognitive",
            "review_count": 138,
            "pct_of_total": 8.0,
            "representative_quotes": [
                {"text": "Spotify has infinite content but zero curation for the moment of 'I want to find something new but don't know what'.", "source": "seed", "rating": 4},
                {"text": "To actually discover new music I have to go to Reddit, read music blogs, ask friends, then manually search Spotify. The app itself is useless for discovery.", "source": "seed", "rating": 2},
                {"text": "There's no 'take me somewhere new' button. The Browse section is algorithmically personalized so it just shows me more of what I know.", "source": "seed", "rating": 3},
            ]
        }
    ],
    "master_synthesis": {
        "master_insight_title": "Reviews Reveal 5 Distinct Discovery Failure Modes, Each With a Different Root Cause and Emotional Signature",
        "connecting_narrative": "The 5 failure modes form a coherent system: the algorithm creates comfort lock (Mode 1), passive listening design removes the motivation to escape it (Mode 2), context-blindness makes even accidental discovery feel wrong (Mode 3), profile fragility punishes any exploration attempt (Mode 4), and the lack of a discovery entry point leaves motivated explorers with nowhere to start (Mode 5). Together, they constitute a discovery-hostile system masquerading as personalization. Spotify has solved relevance at the expense of serendipity — and users feel it.",
        "strategic_implication": "Spotify must decouple 'engagement optimization' from 'discovery enablement' — these are now in direct tension, and resolving it requires a new product surface purpose-built for intentional exploration rather than passive listening.",
        "highest_leverage_mode": "Algorithm Comfort Lock",
        "highest_leverage_rationale": "It affects the largest share of reviews (32%), has the clearest causal link to churn, and is the upstream cause that all other modes amplify. Solving it cascades benefit into Modes 2 and 5."
    },
    "interview_guide": {
        "interview_objective": "Understand the lived experience of discovery failure for Comfort Zone Listeners and validate the 5 failure modes identified in review analysis",
        "screener_criteria": [
            "Spotify Premium subscriber for 2+ years",
            "Listens at least 3x per week",
            "Has noticed their music taste feeling 'stuck' or 'stale' in the past year",
            "Age 25-35, urban setting"
        ],
        "questions": [
            {
                "number": 1,
                "question": "Walk me through the last time you discovered a song or artist that genuinely surprised you — where were you, what were you doing, and how did it happen?",
                "failure_mode_targeted": "Algorithm Comfort Lock",
                "follow_ups": [
                    "Was Spotify involved in that discovery, or did it happen somewhere else?",
                    "When was the last time Spotify itself surprised you in that way?"
                ]
            },
            {
                "number": 2,
                "question": "Tell me about a typical listening session — how does it start, what decisions do you make, and how does it usually end?",
                "failure_mode_targeted": "Passive Listening Drift",
                "follow_ups": [
                    "How often do you actively search for something vs. just letting something play?",
                    "What does it feel like when autoplay takes over?"
                ]
            },
            {
                "number": 3,
                "question": "Think about a moment in the last few months when you really wanted music that matched exactly how you were feeling or what you were doing — what happened when you tried to find it?",
                "failure_mode_targeted": "Context-Blind Recommendations",
                "follow_ups": [
                    "Did Spotify's suggestions feel right for that moment?",
                    "What did you end up doing — finding it, settling, or giving up?"
                ]
            },
            {
                "number": 4,
                "question": "Has there ever been a moment when your Spotify recommendations started feeling 'off' or different than usual — like they stopped understanding you?",
                "failure_mode_targeted": "Profile Contamination Spiral",
                "follow_ups": [
                    "What do you think caused it?",
                    "Did you try to fix it? What happened?"
                ]
            },
            {
                "number": 5,
                "question": "If you wanted to deliberately explore a genre or sound completely outside your usual listening — how would you do that on Spotify today?",
                "failure_mode_targeted": "Discovery Entry Point Void",
                "follow_ups": [
                    "Have you tried? What happened?",
                    "What would the ideal experience look like?"
                ]
            }
        ],
        "closing_question": "If you could add one feature to Spotify that would make you feel like your musical identity is actually growing — not just being fed familiar content — what would it be?"
    },
    "metadata": {
        # NOTE: this demo dataset illustrates the OUTPUT SHAPE of a real
        # run; it does not represent 1,717 actual reviews. The original
        # value here (1717) was a leftover figure from an early test run
        # and is exactly the kind of unverified number app.py's hero
        # section was displaying as "1,700+ reviews analyzed" regardless
        # of whether a real pipeline run had ever produced that count.
        # Demo mode is built from the 25 SEED_REVIEWS; that's the only
        # honest "reviews analyzed" figure for this dataset.
        "total_reviews_analyzed": 25,
        "n_clusters": 5,
        "llm_provider": "Demo mode (no API key set / offline illustration)",
    }
}

# Tag every object in the demo dataset so app.py can render it consistently
# with live-run fallback content — demo mode is intentional and disclosed,
# but it is still NOT genuine LLM synthesis and must never look identical
# to a live result in the UI.
for _fm in DEMO_FAILURE_MODES["failure_modes"]:
    _fm["synthesis_status"] = "demo_mode"
DEMO_FAILURE_MODES["master_synthesis"]["synthesis_status"] = "demo_mode"
DEMO_FAILURE_MODES["interview_guide"]["synthesis_status"] = "demo_mode"
DEMO_FAILURE_MODES["metadata"]["synthesis_status"] = "demo_mode"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(DEMO_FAILURE_MODES["master_synthesis"], indent=2))