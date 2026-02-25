import os
import re
import sqlite3
from dateutil import parser as dtparser
import pandas as pd
from googleapiclient.discovery import build

# ----------------------------
# Config
# ----------------------------
API_KEY = os.getenv("YOUTUBE_API_KEY")  # set in terminal: export YOUTUBE_API_KEY="..."
if not API_KEY:
    raise SystemExit("Missing YOUTUBE_API_KEY env var. Run: export YOUTUBE_API_KEY='YOUR_KEY'")

# Search query to find relevant videos
QUERY = (
    'PrizePicks OR "PrizePicks betting" OR "PrizePicks wins" OR '
    '"PrizePicks losses" OR "PrizePicks experience" OR '
    '"PrizePicks payout" OR "PrizePicks promo code" OR '
    '"PrizePicks review" OR "PrizePicks picks"'
)

MAX_VIDEOS = 40          # number of videos to pull comments from
COMMENTS_PER_VIDEO = 800 # max top-level comments per video (keeps this 1-day sized)

DB_NAME = "prizepicks_youtube.db"
FACTS_CSV = "comment_facts.csv"
DAILY_CSV = "daily_summary.csv"

# ----------------------------
# Simple, explainable tagging
# ----------------------------
THEMES = {
    "trust_payouts": [
        r"\bwithdraw", r"\bpayout", r"\bcash\s*out", r"\bscam", r"\bstole\b",
        r"\brigged\b", r"\bverification\b", r"\blocked\b", r"\bid\b", r"\bban(ned)?\b"
    ],
    "promos_bonuses": [
        r"\bpromo\b", r"\bbonus\b", r"\bfree\b", r"\bdeposit\b", r"\bmatch\b",
        r"\bcode\b", r"\boffer\b", r"\bdiscount\b"
    ],
    "support": [
        r"\bsupport\b", r"\bhelp\b", r"\bemail\b", r"\bcustomer service\b",
        r"\brefund\b", r"\bchargeback\b"
    ],
    "product_ux": [
        r"\bapp\b", r"\bupdate\b", r"\bbug\b", r"\bglitch\b", r"\blag\b",
        r"\bcrash\b", r"\bui\b", r"\binterface\b", r"\blogin\b", r"\bsign\s*in\b"
    ],
    "win_loss_emotion": [
        r"\bwon\b", r"\bwin\b", r"\bhit\b", r"\bcashed\b", r"\bcash\b",
        r"\blost\b", r"\blose\b", r"\bL\b", r"\bprofit\b", r"\bparlay\b"
    ],
}

NEG_WORDS = re.compile(r"\b(sc(am|ammed)|rigged|trash|terrible|worst|hate|broken|refund|stole|locked)\b", re.I)
POS_WORDS = re.compile(r"\b(love|great|good|fire|amazing|legit|easy|smooth)\b", re.I)

def tag_themes(text: str):
    """Return a list of theme tags for a comment."""
    t = (text or "").lower()
    tags = []
    for theme, patterns in THEMES.items():
        if any(re.search(p, t) for p in patterns):
            tags.append(theme)
    return tags or ["other"]

def simple_sentiment(text: str):
    """Lightweight heuristic sentiment: positive/neutral/negative."""
    t = (text or "").lower()
    score = 0
    if NEG_WORDS.search(t):
        score -= 1
    if POS_WORDS.search(t):
        score += 1
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"

# ----------------------------
# YouTube API
# ----------------------------
youtube = build("youtube", "v3", developerKey=API_KEY)

def search_videos(query, max_results=15):
    """Search YouTube for videos matching query."""
    req = youtube.search().list(
        q=query,
        part="snippet",
        type="video",
        maxResults=min(max_results, 50)
    )
    res = req.execute()
    vids = []
    for item in res.get("items", []):
        vids.append({
            "video_id": item["id"]["videoId"],
            "title": item["snippet"]["title"],
            "channel": item["snippet"]["channelTitle"],
            "published_at": item["snippet"]["publishedAt"]
        })
    return vids

def fetch_comments(video_id, max_comments=300):
    """Fetch top-level comments for a video."""
    comments = []
    page_token = None

    while len(comments) < max_comments:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=page_token,
            textFormat="plainText"
        )
        res = req.execute()

        for item in res.get("items", []):
            sn = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["id"],
                "video_id": video_id,
                "author": sn.get("authorDisplayName"),
                "published_at": sn.get("publishedAt"),
                "like_count": sn.get("likeCount", 0),
                "text": sn.get("textDisplay", "")
            })
            if len(comments) >= max_comments:
                break

        page_token = res.get("nextPageToken")
        if not page_token:
            break

    return comments

# ----------------------------
# Main pipeline
# ----------------------------
def main():
    videos = search_videos(QUERY, MAX_VIDEOS)
    if not videos:
        raise SystemExit("No videos found. Try changing the QUERY string.")

    all_rows = []
    for v in videos:
        rows = fetch_comments(v["video_id"], COMMENTS_PER_VIDEO)
        for r in rows:
            r["video_title"] = v["title"]
            r["channel"] = v["channel"]
            r["video_published_at"] = v["published_at"]
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise SystemExit("No comments pulled. Try increasing MAX_VIDEOS or adjusting the QUERY.")

    # Time + features
    df["published_dt"] = df["published_at"].apply(lambda x: dtparser.parse(x))
    df["date"] = df["published_dt"].dt.date.astype(str)
    df["sentiment"] = df["text"].apply(simple_sentiment)
    df["theme"] = df["text"].apply(tag_themes)

    # explode theme list -> one row per theme tag
    df = df.explode("theme").reset_index(drop=True)

    # Save to SQLite
    conn = sqlite3.connect(DB_NAME)
    df.to_sql("comment_facts", conn, if_exists="replace", index=False)

    # Daily summary (Tableau-friendly)
    daily = (
        df.groupby(["date", "theme", "sentiment"])
          .size()
          .reset_index(name="comment_count")
    )
    daily.to_sql("daily_summary", conn, if_exists="replace", index=False)

    # Export CSVs (easy for Tableau)
    df.to_csv(FACTS_CSV, index=False)
    daily.to_csv(DAILY_CSV, index=False)

    print(f"Done. Files created: {DB_NAME}, {FACTS_CSV}, {DAILY_CSV}")
    print(f"Rows: comment_facts={len(df):,} | daily_summary={len(daily):,}")
    print("Next: open Tableau and connect to daily_summary.csv (and optionally comment_facts.csv).")

if __name__ == "__main__":
    main()