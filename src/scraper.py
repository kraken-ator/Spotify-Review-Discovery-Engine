"""
scraper.py — Review Ingestion Layer
Sources:
  1. App Store (iOS)          — Apple RSS feed, keyword-filtered post-fetch
  2. Google Play Store        — google-play-scraper, keyword-filtered post-fetch
  3. Reddit r/spotify         — Apify JSON export (drop file into data/)
  4. Spotify Community Forums — requests + BeautifulSoup, query-scoped
  5. Twitter/X                — Tweepy v2 search (bearer token), query-scoped
  6. Seed reviews             — 25 synthetic placeholder reviews, OFF by
                                 default. Every review carries a
                                 `provenance` field ("scraped" vs.
                                 "synthetic_seed") so seed content is never
                                 silently blended into a real-data count.
                                 Turn on only for offline demo purposes when
                                 no live sources are reachable.

Output: unified DataFrame → data/raw_reviews.csv
"""

import os
import json
import time
import logging
import re
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Discovery-Relevance Filter (shared by all sources)
# ─────────────────────────────────────────────
# Community Forums and Twitter were already scoped to discovery-specific
# queries at fetch time. App Store and Play Store were not — they pulled
# "most recent" / "most relevant" reviews with no topic filter, which is
# why the dominant cluster in a live run was a freemium/ads complaint
# (the single most common topic in *any* ad-supported app's review
# stream) instead of anything about discovery. This filter closes that
# gap by applying the same keyword scoping to every source post-fetch.
#
# Keyword matching (vs. an LLM relevance classifier) is the right tool
# here: it's auditable — anyone reviewing this code can see exactly
# which terms qualify a review — and it costs zero API calls, which
# matters when the synthesis layer is already rate-limited.

DISCOVERY_KEYWORDS = [
    # direct discovery language
    "discover", "discovery", "recommend", "recommendation", "algorithm",
    "suggest", "suggestion",
    # named features that ARE the discovery surface
    "discover weekly", "daily mix", "release radar", "radio", "blend",
    "wrapped",
    # the comfort-loop / repetition complaint (core to this project's thesis)
    "same song", "same songs", "same artist", "same music", "repetitive",
    "repeat", "stale", "stuck in a loop", "comfort zone", "bubble",
    "familiar", "predictable", "boring playlist",
    # exploration intent / new-music language
    "new music", "new artist", "new artists", "new song", "new band",
    "explore", "exploration", "browse", "genre", "taste", "personali",
    "similar artist", "shuffle algorithm",
]


def is_discovery_relevant(text: str, title: str = "") -> bool:
    """
    Keyword relevance gate applied to App Store / Play Store reviews.
    Returns True if the review text or title contains any discovery-
    or recommendation-related language. This is the post-filter the
    fellowship evaluation recommended in place of unfiltered "most
    recent" scraping.
    """
    combined = f"{title} {text}".lower()
    return any(kw in combined for kw in DISCOVERY_KEYWORDS)


# ─────────────────────────────────────────────
# 1. App Store (iOS) - Official RSS Fallback
# ─────────────────────────────────────────────

def fetch_appstore_reviews(
    app_id: str = "324684580",
    max_reviews: int = 500,
    filter_discovery: bool = True,
) -> list[dict]:
    """
    Fetch iOS App Store reviews via Apple's public RSS JSON feed across multiple countries
    to bypass the 500-review per-country hard limit.
    """
    try:
        import requests
    except ImportError:
        log.warning("requests not installed. Run: pip install requests")
        return []

    reviews = []
    # Expand raw pool by hitting multiple English-speaking storefronts
    countries = ["us", "gb", "ca", "au", "in", "ie", "nz", "za", "sg", "ph"]
    fetch_target = 500 if filter_discovery else max_reviews

    log.info(f"Fetching App Store reviews across {len(countries)} countries to bypass Apple's 500-review limit...")

    for country in countries:
        if len(reviews) >= fetch_target * len(countries):
            break

        pages = min(10, (fetch_target // 50) + 1)
        for page in range(1, pages + 1):
            url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"

            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code != 200:
                    break

                data = resp.json()
                entries = data.get("feed", {}).get("entry", [])

                if not entries:
                    break

                for entry in entries:
                    # Skip the first entry if it's just app metadata instead of a user review
                    if "author" not in entry:
                        continue

                    reviews.append({
                        "source": "app_store",
                        "review_id": f"{country}_{entry.get('id', {}).get('label', '')}",
                        "rating": int(entry.get("im:rating", {}).get("label", 0)),
                        "title": entry.get("title", {}).get("label", ""),
                        "text": entry.get("content", {}).get("label", ""),
                        "date": entry.get("updated", {}).get("label", ""),
                        "helpful_votes": 0, # RSS feed does not provide helpful votes
                    })

            except Exception as e:
                log.debug(f"App Store RSS fetch failed on {country} page {page}: {e}")
                break

            time.sleep(0.5)  # Be polite to Apple's servers

    log.info(f"  → {len(reviews)} App Store reviews fetched (raw, pre-filter, across {len(countries)} countries)")

    if filter_discovery:
        relevant = [r for r in reviews if is_discovery_relevant(r["text"], r.get("title", ""))]
        pct = (len(relevant) / len(reviews) * 100) if reviews else 0.0
        log.info(
            f"  → {len(relevant)}/{len(reviews)} ({pct:.1f}%) are discovery-relevant "
            f"after keyword filtering"
        )
        reviews = relevant

    return reviews[:max_reviews]


# ─────────────────────────────────────────────
# 2. Google Play Store
# ─────────────────────────────────────────────

def fetch_playstore_reviews(
    app_id: str = "com.spotify.music",
    max_reviews: int = 500,
    filter_discovery: bool = True,
    oversample_factor: int = 8,
    max_raw_pool: int = 5000,
) -> list[dict]:
    """
    Fetch Android Play Store reviews for Spotify.

    google-play-scraper's reviews() has no query/search parameter either
    (same limitation as the App Store RSS feed), so discovery relevance
    is enforced as a post-filter here too. Unlike the App Store's hard
    500-review RSS ceiling, this API can return a much larger raw pool,
    so when filtering we deliberately oversample the raw fetch (default
    8x the requested count, capped at max_raw_pool) to make sure enough
    discovery-relevant reviews survive the keyword filter.
    """
    try:
        from google_play_scraper import reviews as gp_reviews, Sort
    except ImportError:
        log.warning("google-play-scraper not installed. Run: pip install google-play-scraper")
        return []

    fetch_target = min(max_reviews * oversample_factor, max_raw_pool) if filter_discovery else max_reviews

    log.info(f"Fetching Play Store reviews (raw pool target={fetch_target})...")
    result, _ = gp_reviews(
        app_id,
        lang="en",
        country="us",
        sort=Sort.MOST_RELEVANT,
        count=fetch_target,
    )

    reviews = []
    for r in result:
        reviews.append({
            "source": "play_store",
            "review_id": r.get("reviewId", ""),
            "rating": r.get("score", None),
            "title": "",
            "text": r.get("content", ""),
            "date": str(r.get("at", "")),
            "helpful_votes": r.get("thumbsUpCount", 0),
        })
    log.info(f"  → {len(reviews)} Play Store reviews fetched (raw, pre-filter)")

    if filter_discovery:
        relevant = [r for r in reviews if is_discovery_relevant(r["text"], r.get("title", ""))]
        pct = (len(relevant) / len(reviews) * 100) if reviews else 0.0
        log.info(
            f"  → {len(relevant)}/{len(reviews)} ({pct:.1f}%) are discovery-relevant "
            f"after keyword filtering"
        )
        reviews = relevant
        if len(reviews) < max_reviews:
            log.warning(
                f"  ⚠ Only {len(reviews)} discovery-relevant Play Store reviews found "
                f"after oversampling {fetch_target} raw reviews. Consider raising "
                f"oversample_factor or max_raw_pool if you need more."
            )

    return reviews[:max_reviews]


# ─────────────────────────────────────────────
# 3. Reddit — Apify JSON export
# ─────────────────────────────────────────────

def load_apify_reddit(file_path: str = None, max_posts: int = 1000) -> list[dict]:
    """
    Load Reddit data from an Apify JSON export.

    Apify's Reddit Scraper exports an array of objects. Common schemas:
      - {id, title, selftext, score, url, created_utc, subreddit, comments:[{body,score}]}
      - {postId, postTitle, body, upvotes, createdAt, commentsData:[{commentBody}]}
    This loader handles both schemas and any field-name variations.

    HOW TO USE:
      1. Run the Apify Reddit Scraper actor on r/spotify with your keywords
      2. Export dataset as JSON
      3. Save the file to: data/apify_reddit.json  (or set APIFY_REDDIT_FILE in .env)
      4. Run the pipeline — it will auto-detect and parse the file
    """
    # Resolve file path
    if not file_path:
        file_path = os.getenv("APIFY_REDDIT_FILE", "data/apify_reddit.json")

    path = Path(file_path)
    if not path.exists():
        log.warning(f"Apify Reddit file not found at '{file_path}'.")
        log.warning("To use Reddit data: export from Apify → save as data/apify_reddit.json")
        return []

    log.info(f"Loading Apify Reddit export from {file_path}...")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Handle wrapped exports: {"data": [...]} or {"items": [...]}
    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("items") or raw.get("results") or list(raw.values())[0]
    if not isinstance(raw, list):
        log.warning("Unrecognized Apify JSON structure. Expected a list of posts.")
        return []

    reviews = []
    seen_ids = set()

    for item in raw[:max_posts * 3]:  # over-fetch, will trim after dedup
        if len(reviews) >= max_posts:
            break

        # ── Normalise field names across Apify schema variants ──
        post_id   = str(item.get("id") or item.get("postId") or item.get("post_id") or "")
        title     = item.get("title") or item.get("postTitle") or item.get("post_title") or ""
        body      = item.get("selftext") or item.get("body") or item.get("text") or item.get("content") or ""
        score     = item.get("score") or item.get("upvotes") or item.get("ups") or 0
        created   = item.get("created_utc") or item.get("createdAt") or item.get("created") or ""
        subreddit = item.get("subreddit") or item.get("subredditName") or "spotify"
        url       = item.get("url") or item.get("postUrl") or ""

        # Parse date
        date_str = ""
        if isinstance(created, (int, float)):
            date_str = str(datetime.fromtimestamp(created, tz=timezone.utc))
        elif isinstance(created, str):
            date_str = created

        # Add the post body if substantive
        if body and len(body.strip()) >= 60 and post_id not in seen_ids:
            seen_ids.add(post_id)
            reviews.append({
                "source": "reddit",
                "review_id": f"reddit_post_{post_id}",
                "rating": None,
                "title": title,
                "text": body.strip()[:2000],
                "date": date_str,
                "helpful_votes": int(score) if score else 0,
            })

        # Add comments
        comments = (
            item.get("comments") or
            item.get("commentsData") or
            item.get("topComments") or []
        )
        for c in comments:
            c_id   = str(c.get("id") or c.get("commentId") or "")
            c_body = c.get("body") or c.get("commentBody") or c.get("text") or ""
            c_score = c.get("score") or c.get("upvotes") or 0
            c_created = c.get("created_utc") or c.get("createdAt") or ""

            c_date = ""
            if isinstance(c_created, (int, float)):
                c_date = str(datetime.fromtimestamp(c_created, tz=timezone.utc))
            elif isinstance(c_created, str):
                c_date = c_created

            uid = f"reddit_comment_{c_id or c_body[:30]}"
            if c_body and len(c_body.strip()) >= 60 and uid not in seen_ids:
                seen_ids.add(uid)
                reviews.append({
                    "source": "reddit_comment",
                    "review_id": uid,
                    "rating": None,
                    "title": title,  # parent post title as context
                    "text": c_body.strip()[:1500],
                    "date": c_date,
                    "helpful_votes": int(c_score) if c_score else 0,
                })

    log.info(f"  → {len(reviews)} Reddit items loaded from Apify export")
    return reviews[:max_posts]


# ─────────────────────────────────────────────
# 4. Spotify Community Forums
# ─────────────────────────────────────────────

COMMUNITY_SEARCH_QUERIES = [
    "discover weekly recommendation",
    "algorithm same songs",
    "new music discovery",
    "repeat songs loop",
    "music taste bubble",
    "explore new artists",
]

COMMUNITY_BASE = "https://community.spotify.com"
COMMUNITY_SEARCH = "https://community.spotify.com/t5/forums/searchpage/tab/message?q={query}&search_type=thread"

def fetch_community_forums(max_posts: int = 200) -> list[dict]:
    """
    Scrape Spotify Community Forums (community.spotify.com) for discovery-related threads.
    Uses requests + BeautifulSoup. No auth required — public content.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("requests/beautifulsoup4 not installed.")
        return []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; SpotifyPMResearch/1.0)",
        "Accept-Language": "en-US,en;q=0.9",
    }

    reviews = []
    seen_urls = set()

    for query in tqdm(COMMUNITY_SEARCH_QUERIES, desc="Community Forums"):
        if len(reviews) >= max_posts:
            break
        try:
            search_url = COMMUNITY_SEARCH.format(query=query.replace(" ", "+"))
            resp = requests.get(search_url, headers=headers, timeout=15)
            if resp.status_code != 200:
                log.warning(f"Community search returned {resp.status_code} for '{query}'")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Thread links — Lithium CMS structure
            thread_links = []
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if "/td-p/" in href or "/m-p/" in href:
                    full = href if href.startswith("http") else COMMUNITY_BASE + href
                    if full not in seen_urls:
                        thread_links.append(full)
                        seen_urls.add(full)

            for url in thread_links[:5]:
                try:
                    time.sleep(0.8)
                    t_resp = requests.get(url, headers=headers, timeout=15)
                    if t_resp.status_code != 200:
                        continue
                    t_soup = BeautifulSoup(t_resp.text, "html.parser")

                    # Extract post bodies
                    post_divs = t_soup.find_all("div", class_=re.compile(r"lia-message-body|post-body|message-content"))
                    title_tag = t_soup.find("h1") or t_soup.find(class_=re.compile(r"lia-message-subject|thread-title"))
                    title_text = title_tag.get_text(strip=True) if title_tag else ""

                    for i, div in enumerate(post_divs[:6]):
                        text = div.get_text(separator=" ", strip=True)
                        text = re.sub(r"\s+", " ", text).strip()
                        if len(text) >= 80:
                            reviews.append({
                                "source": "spotify_community",
                                "review_id": f"community_{hash(text[:60])}",
                                "rating": None,
                                "title": title_text if i == 0 else f"Re: {title_text}",
                                "text": text[:2000],
                                "date": "",
                                "helpful_votes": 0,
                            })
                except Exception as e:
                    log.debug(f"Thread fetch error: {e}")
                    continue

            time.sleep(1.2)

        except Exception as e:
            log.warning(f"Community search error for '{query}': {e}")
            continue

    log.info(f"  → {len(reviews)} Spotify Community posts fetched")
    return reviews[:max_posts]


# ─────────────────────────────────────────────
# 5a. Twitter/X
# ─────────────────────────────────────────────

TWITTER_QUERIES = [
    "spotify discover weekly -is:retweet lang:en",
    "spotify algorithm same songs -is:retweet lang:en",
    "spotify music discovery boring -is:retweet lang:en",
    "spotify recommendations suck -is:retweet lang:en",
    "spotify listening bubble -is:retweet lang:en",
]

def fetch_twitter_posts(max_tweets: int = 300) -> list[dict]:
    """
    Fetch tweets about Spotify discovery using Tweepy v2 (bearer token).
    Requires a Twitter Developer account with Basic or Pro access.
    Free tier: 500k tweets/month read with Basic ($100/mo).
    Academic/Essential: limited to 10 recent results per query without elevated access.

    If bearer token is missing, this source is skipped gracefully.
    """
    bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer_token or bearer_token == "your-twitter-bearer-token-here":
        log.warning("TWITTER_BEARER_TOKEN not set. Skipping Twitter source.")
        log.warning("Get one at: developer.twitter.com → Projects → Your App → Keys & Tokens")
        return []

    try:
        import tweepy
    except ImportError:
        log.warning("tweepy not installed. Run: pip install tweepy")
        return []

    log.info(f"Fetching tweets (target={max_tweets})...")
    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

    reviews = []
    seen_ids = set()
    per_query = max(10, max_tweets // len(TWITTER_QUERIES))

    # Date filter: last 7 days (v2 recent search free tier limit)
    start_time = datetime.now(tz=timezone.utc) - timedelta(days=7)

    for query in TWITTER_QUERIES:
        if len(reviews) >= max_tweets:
            break
        try:
            resp = client.search_recent_tweets(
                query=query,
                max_results=min(per_query, 100),
                tweet_fields=["created_at", "public_metrics", "text", "id"],
                start_time=start_time,
            )
            if not resp.data:
                continue

            for tweet in resp.data:
                if tweet.id in seen_ids:
                    continue
                seen_ids.add(tweet.id)
                metrics = tweet.public_metrics or {}
                reviews.append({
                    "source": "twitter",
                    "review_id": f"tweet_{tweet.id}",
                    "rating": None,
                    "title": "",
                    "text": tweet.text,
                    "date": str(tweet.created_at),
                    "helpful_votes": metrics.get("like_count", 0) + metrics.get("retweet_count", 0),
                })

            time.sleep(1.5)

        except tweepy.TweepyException as e:
            log.warning(f"Twitter API error: {e}")
            # Check for rate limit or auth issues specifically
            if "403" in str(e) or "401" in str(e):
                log.warning("Twitter auth failed. Check your bearer token and API access level.")
                break
            continue

    log.info(f"  → {len(reviews)} tweets fetched")
    return reviews[:max_tweets]

# ─────────────────────────────────────────────
# 5b. Twitter/X — Apify JSON export
# ─────────────────────────────────────────────

def load_apify_twitter(file_path: str = "data/apify_twitter.json", max_tweets: int = 1000) -> list[dict]:
    """Load Twitter data from an Apify JSON export."""
    path = Path(file_path)
    if not path.exists():
        log.warning(f"Apify Twitter file not found at '{file_path}'.")
        return []

    log.info(f"Loading Apify Twitter export from {file_path}...")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        raw = raw.get("data") or raw.get("items") or raw.get("results") or list(raw.values())[0]

    reviews = []
    seen_ids = set()

    for item in raw:
        if len(reviews) >= max_tweets:
            break

        # Handle different Apify scraper output formats
        t_id = str(item.get("id") or item.get("id_str") or "")
        text = item.get("full_text") or item.get("text") or ""
        created = item.get("created_at") or item.get("createdAt") or ""
        likes = item.get("favorite_count") or item.get("likes") or item.get("like_count") or 0
        retweets = item.get("retweet_count") or item.get("retweets") or 0

        # Sometimes text is nested inside a 'tweet' object
        if not text and item.get("tweet"):
            text = item["tweet"].get("full_text") or item["tweet"].get("text") or ""

        if text and t_id not in seen_ids:
            seen_ids.add(t_id)
            reviews.append({
                "source": "twitter_apify",
                "review_id": f"tweet_{t_id}",
                "rating": None,
                "title": "",
                "text": text.strip(),
                "date": str(created),
                "helpful_votes": int(likes) + int(retweets),
            })

    log.info(f"  → {len(reviews)} Twitter items loaded from Apify export")
    return reviews[:max_tweets]
# ─────────────────────────────────────────────
# 6. Seed Reviews (built-in fallback)
# ─────────────────────────────────────────────

SEED_REVIEWS = [
    # Comfort-loop / algorithm fatigue
    {"source": "seed", "review_id": "s001", "rating": 3, "title": "Same songs forever", "text": "I've been using Spotify for 4 years and Discover Weekly keeps recommending me the same artists I already follow. I added one Taylor Swift song and now that's all I get. It's like the algorithm is allergic to anything new.", "date": "2024-01-15", "helpful_votes": 203},
    {"source": "seed", "review_id": "s002", "rating": 2, "title": "Stuck in a bubble", "text": "The recommendations have gotten worse over time. I feel completely trapped in a listening bubble. Every radio station sounds identical. I went from discovering 10 new artists a month to basically zero.", "date": "2024-02-03", "helpful_votes": 189},
    {"source": "seed", "review_id": "s003", "rating": 4, "title": "Discovery Weekly used to be great", "text": "Discover Weekly was amazing when I first started using Spotify. Now it's just variations of stuff I already like. I want to be surprised, not just validated. It's like the algorithm learned what I like then stopped trying.", "date": "2024-01-28", "helpful_votes": 445},
    {"source": "seed", "review_id": "s004", "rating": 2, "title": "Algorithm killed my music taste", "text": "I genuinely feel like Spotify has narrowed my taste, not expanded it. I listen to the same 200 songs. I know that's on me partly but the algorithm just feeds the loop. I've started using YouTube for actual discovery.", "date": "2024-03-10", "helpful_votes": 312},
    {"source": "seed", "review_id": "s005", "rating": 3, "title": "No serendipity anymore", "text": "Remember when you'd hear a random song on the radio and it'd change your life? Spotify used to give me that feeling occasionally. Now it's just a very sophisticated version of playing my own library back at me.", "date": "2024-02-20", "helpful_votes": 267},
    # Autoplay / passive listening trap
    {"source": "seed", "review_id": "s006", "rating": 2, "title": "Autoplay is a black hole", "text": "I put on a playlist, it ends, autoplay kicks in and suddenly I'm three hours deep in generic acoustic chill music I never chose. I didn't consent to this. I wanted to discover music, not be spoon-fed Spotify's playlist library.", "date": "2024-01-05", "helpful_votes": 178},
    {"source": "seed", "review_id": "s007", "rating": 3, "title": "Passive listening is too easy", "text": "The app makes it so easy to just let music play that I never actively choose anything anymore. My Wrapped showed I listened to 40,000 minutes and discovered 3 new artists. Three. In a year.", "date": "2024-01-10", "helpful_votes": 521},
    {"source": "seed", "review_id": "s008", "rating": 2, "title": "Radio ruins discovery", "text": "Spotify Radio is the worst feature for discovery. It plays the same 30 songs on rotation. I've heard the same 'similar artist' recommendation 47 times. It's not radio, it's a very slow shuffle.", "date": "2024-02-14", "helpful_votes": 234},
    # Context mismatch
    {"source": "seed", "review_id": "s009", "rating": 3, "title": "Doesn't know my context", "text": "I listen to focus music at work, party playlists on weekends, sad songs when I'm emotional. Spotify has no idea what I need right now — it just plays based on what I listened to yesterday. I need music that fits the moment, not my history.", "date": "2024-03-01", "helpful_votes": 389},
    {"source": "seed", "review_id": "s010", "rating": 2, "title": "Wrong vibe every time", "text": "I'll be in the mood to explore something completely different and Spotify will serve me exactly what I always hear. There's no way to say 'I want something I've never heard, in a direction I haven't gone, show me something weird and wonderful'.", "date": "2024-02-08", "helpful_votes": 156},
    {"source": "seed", "review_id": "s011", "rating": 4, "title": "Great for lazy days, bad for growth", "text": "Perfect for background music when I don't care. Completely useless when I actually want to find something new and meaningful. The two use cases need completely different UX and Spotify only solves for one.", "date": "2024-01-22", "helpful_votes": 298},
    # Playlist contamination
    {"source": "seed", "review_id": "s012", "rating": 2, "title": "One bad stream poisoned everything", "text": "I played a podcast for someone else in the car. Now my entire algorithmic profile is contaminated. Daily Mix started including true crime podcasts. Discover Weekly gave me crime drama soundtracks. No way to undo this.", "date": "2024-02-28", "helpful_votes": 445},
    {"source": "seed", "review_id": "s013", "rating": 3, "title": "Shared account problem", "text": "My partner and I share an account. The algorithm is now a confused mess that doesn't know if it's recommending K-pop or death metal. We literally can't use any algorithmic features anymore. There's no solution for this.", "date": "2024-01-18", "helpful_votes": 367},
    {"source": "seed", "review_id": "s014", "rating": 2, "title": "Gym music invaded everything", "text": "I created a high-energy workout playlist. Now EVERY recommendation is either EDM or hip-hop regardless of context. My jazz recommendations disappeared. My folk recommendations gone. One playlist wrecked six months of algorithmic learning.", "date": "2024-03-05", "helpful_votes": 189},
    # Discovery entry void
    {"source": "seed", "review_id": "s015", "rating": 4, "title": "Too much choice paralysis", "text": "When I try to explore Browse or Search, I'm overwhelmed by 80 million songs and don't know where to start. I just go back to my saved songs. Spotify has infinite content but zero curation for the moment of 'I want to find something new but don't know what'.", "date": "2024-01-30", "helpful_votes": 312},
    {"source": "seed", "review_id": "s016", "rating": 3, "title": "Where do I even start?", "text": "I want to explore a new genre I've never listened to. How do I do that on Spotify? I genuinely don't know. The Browse section is algorithmically personalized so it just shows me more of what I know. There's no 'take me somewhere new' button.", "date": "2024-02-15", "helpful_votes": 278},
    {"source": "seed", "review_id": "s017", "rating": 2, "title": "Discovery requires too much work", "text": "To actually discover new music I have to go to Reddit, read music blogs, ask friends, then manually search Spotify. The app itself is useless for discovery unless you already know what you want to find. That's backwards.", "date": "2024-02-22", "helpful_votes": 356},
    # Social / identity
    {"source": "seed", "review_id": "s018", "rating": 3, "title": "I used to have interesting taste", "text": "My Spotify Wrapped used to surprise me. I'd discover I'd gone deep on some artist I barely remembered. Now it's just the same 10 artists I've liked since college. The algorithm is making me a less interesting listener.", "date": "2024-01-08", "helpful_votes": 423},
    {"source": "seed", "review_id": "s019", "rating": 2, "title": "My taste feels frozen in time", "text": "I'm 31 and my Spotify profile looks like I'm 22. It still thinks I love the bands I listened to obsessively in college. It hasn't noticed that my tastes have evolved. There's no way to signal 'I've grown, show me something that matches who I am now'.", "date": "2024-02-11", "helpful_votes": 534},
    {"source": "seed", "review_id": "s020", "rating": 4, "title": "Friends recommend better music than the algorithm", "text": "Every significant music discovery in the last 2 years came from a friend, not Spotify. The algorithm is technically impressive but culturally empty. It can match audio signatures but it can't understand why a song matters at a specific moment in your life.", "date": "2024-03-08", "helpful_votes": 612},
    {"source": "seed", "review_id": "s021", "rating": 3, "title": "Discover Weekly regression", "text": "There was a period around 2019-2020 where Discover Weekly was almost magical. It found artists I'd never heard that became favorites. Something changed. The playlists are safer now. More mainstream. Less weird. I miss the algorithm that took risks.", "date": "2024-01-25", "helpful_votes": 289},
    {"source": "seed", "review_id": "s022", "rating": 2, "title": "Can't escape the comfort zone algorithmically", "text": "I made a deliberate effort to listen to unfamiliar genres for a month. Brazilian jazz, Mongolian folk, experimental electronic. After 30 days of effort, Discover Weekly gave me... indie folk. My usual stuff. The algorithm doesn't reward exploration.", "date": "2024-02-05", "helpful_votes": 478},
    {"source": "seed", "review_id": "s023", "rating": 3, "title": "The skip tax is real", "text": "If I skip an unfamiliar song, the algorithm learns I don't like that direction. But I skip because I'm not in the mood RIGHT NOW, not because I hate it forever. There's no way to say 'try again later'. Every skip is permanent retraining against discovery.", "date": "2024-02-19", "helpful_votes": 334},
    {"source": "seed", "review_id": "s024", "rating": 4, "title": "I want to be challenged", "text": "The best musical discoveries I've made required some initial resistance — a song I didn't immediately like that I gave a chance. Spotify's algorithm only serves things I'll like immediately. It never challenges me, never pushes, never says 'stick with this, it'll grow on you'.", "date": "2024-03-12", "helpful_votes": 567},
    {"source": "seed", "review_id": "s025", "rating": 2, "title": "Algorithm is too safe", "text": "I'm a music journalist. My job requires discovering new music constantly. Spotify is almost useless for professional discovery. It's optimized for engagement, not exploration. I have to use Bandcamp, SoundCloud, and Resident Advisor to actually find new artists.", "date": "2024-01-14", "helpful_votes": 445},
]


# ─────────────────────────────────────────────
# Master Ingestion
# ─────────────────────────────────────────────

def ingest_all_reviews(
    use_appstore: bool = True,
    use_playstore: bool = True,
    use_reddit_apify: bool = True,
    use_twitter_apify: bool = True,
    use_community: bool = True,
    use_twitter: bool = True,
    use_seed: bool = False,
    filter_discovery: bool = True,
    max_per_source: int = 500,
    apify_reddit_file: str = None,
    apify_twitter_file: str = "data/apify_twitter.json",
    output_path: str = "data/raw_reviews.csv",
) -> pd.DataFrame:
    """
    Ingest reviews from all sources and return a unified cleaned DataFrame.

    use_seed defaults to False. SEED_REVIEWS are 25 synthetic placeholder
    reviews written during early scaffolding — useful for offline demos
    when no live source is reachable, but they must never be silently
    blended into a "reviews analyzed" headline without disclosure. Every
    review in the returned DataFrame carries a `provenance` column
    ("scraped" or "synthetic_seed") so this is always auditable downstream,
    regardless of how use_seed is set.

    filter_discovery is passed through to the App Store and Play Store
    fetchers, which have no native query/search parameter and therefore
    need a post-fetch keyword filter to stay on-topic (see
    is_discovery_relevant). Reddit, Community Forums, and Twitter are
    already scoped to discovery-specific queries at fetch time.
    """
    all_reviews = []

    if use_seed:
        all_reviews.extend(SEED_REVIEWS)
        log.info(f"Loaded {len(SEED_REVIEWS)} synthetic seed reviews (offline-demo content, not live data)")

    if use_appstore:
        try:
            all_reviews.extend(fetch_appstore_reviews(max_reviews=max_per_source, filter_discovery=filter_discovery))
        except Exception as e:
            log.warning(f"App Store scrape failed: {e}")

    if use_playstore:
        try:
            all_reviews.extend(fetch_playstore_reviews(max_reviews=max_per_source, filter_discovery=filter_discovery))
        except Exception as e:
            log.warning(f"Play Store scrape failed: {e}")

    if use_reddit_apify:
        try:
            all_reviews.extend(load_apify_reddit(file_path=apify_reddit_file, max_posts=max_per_source))
        except Exception as e:
            log.warning(f"Apify Reddit load failed: {e}")

    if use_twitter_apify:
        try:
            all_reviews.extend(load_apify_twitter(file_path=apify_twitter_file, max_tweets=max_per_source))
        except Exception as e:
            log.warning(f"Apify Twitter load failed: {e}")

    if use_community:
        try:
            all_reviews.extend(fetch_community_forums(max_posts=max_per_source))
        except Exception as e:
            log.warning(f"Spotify Community scrape failed: {e}")

    if use_twitter:
        try:
            all_reviews.extend(fetch_twitter_posts(max_tweets=max_per_source))
        except Exception as e:
            log.warning(f"Twitter fetch failed: {e}")

    if not all_reviews:
        log.error("No reviews collected from any source. Check your credentials and network.")
        return pd.DataFrame()

    df = pd.DataFrame(all_reviews)

    # ── Provenance tagging ──
    # Single point of truth: "seed" is the only source value that maps to
    # synthetic content. Tagging here (rather than at each fetch site) means
    # this can never drift out of sync with an individual scraper.
    df["provenance"] = df["source"].apply(lambda s: "synthetic_seed" if s == "seed" else "scraped")

    # ── Cleaning ──
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() >= 50]
    df = df.drop_duplicates(subset=["text"])
    df["word_count"] = df["text"].str.split().str.len()
    df = df[df["word_count"] >= 10]
    df = df.reset_index(drop=True)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    n_seed = int((df["provenance"] == "synthetic_seed").sum())
    n_scraped = int((df["provenance"] == "scraped").sum())

    log.info(f"\n✓ Total reviews saved: {len(df)} → {output_path}")
    log.info(f"  Provenance: {n_scraped} scraped (real) + {n_seed} synthetic seed")
    log.info(f"  Source breakdown: {df['source'].value_counts().to_dict()}")
    if n_seed > 0:
        log.warning(
            f"  ⚠ {n_seed} synthetic seed reviews are included in this run. "
            f"Disclose this count wherever total review figures are reported."
        )

    return df


if __name__ == "__main__":
    df = ingest_all_reviews()
    print(f"\n{'='*50}")
    print(f"Total reviews: {len(df)}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    rated = df["rating"].dropna()
    if len(rated):
        print(f"Avg rating: {rated.mean():.2f}")
    print(df.head(3)[["source", "rating", "text"]].to_string())
