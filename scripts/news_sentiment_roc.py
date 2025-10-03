#!/usr/bin/env python3
"""
Simplified Roc Nation News Sentiment with robust themes.

Input (exactly one row per Name):
  roc_aliases.csv  -> columns: brand,alias   # brand=name, alias=Context

Outputs:
  data_roc/articles/YYYY-MM-DD.csv
  data_roc/daily_counts.csv   (includes 'company' and a short 'theme')

Query used per name:
  "<n>" "<Context>"

Theme logic (negative headlines only):
  1) bigram/trigram with min_df>=2
  2) fallback bigram with min_df>=1
  3) fallback unigrams (top 2–3 words)
All while removing stopwords + CEO/Company tokens + "ceo".
"""

import os, csv, time, math, re
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from urllib.parse import urlencode, urlparse

import feedparser
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# ------------------ Config ------------------
HL   = "en-US"
GL   = "US"
CEID = "US:en"

# Roc inputs
BRANDS_TXT = "roc.txt"
ALIASES_CSV = "roc_aliases.csv"

# Output base folder (isolated from other pipelines)
OUTPUT_BASE = "data_roc"
ARTICLES_DIR = os.path.join(OUTPUT_BASE, "articles")
COUNTS_CSV  = os.path.join(OUTPUT_BASE, "daily_counts.csv")

REQUEST_PAUSE_SEC     = 1.0   # polite delay between CEOs
MAX_ITEMS_PER_QUERY   = 40    # cap per CEO per day
PURGE_OLDER_THAN_DAYS = 90

MAX_THEME_WORDS = 10

BASE_STOPWORDS = set((
    "the","a","an","and","or","but","of","for","to","in","on","at","by","with","from","as","about","after","over","under",
    "this","that","these","those","it","its","their","his","her","they","we","you","our","your","i",
    "is","are","was","were","be","been","being","has","have","had","do","does","did","will","would","should","can","could","may","might","must",
    "new","update","updates","report","reports","reported","says","say","said","see","sees","seen","watch","market","stock","shares","share","price","prices",
    "wins","loss","losses","gain","gains","up","down","amid","amidst","news","today","latest","analyst","analysts","rating","cut","cuts","downgrade","downgrades",
    "quarter","q1","q2","q3","q4","year","yrs","2024","2025","2026","usd","billion","million","percent","pct","vs","inc","corp","co","ltd","plc"
))

# Filter out low-signal sources
BLOCKED_DOMAINS = {
    "www.prnewswire.com",
    "www.businesswire.com",
    "www.globenewswire.com",
    "investorplace.com",
    "seekingalpha.com",
}

# ------------------ Helpers ------------------

def today_str() -> str:
    return datetime.now(ZoneInfo("US/Eastern")).strftime("%Y-%m-%d")

def load_ceo_company_map(path: str) -> dict:
    """Read ceo_aliases.csv (brand=CEO, alias=Company) -> {CEO: Company}"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            ceo = (row.get("brand") or "").strip()
            company = (row.get("alias") or "").strip()
            if ceo and company:
                out[ceo] = company
    return out

def google_news_rss_url(query: str) -> str:
    base = "https://news.google.com/rss/search"
    params = {"q": query, "hl": HL, "gl": GL, "ceid": CEID}
    return base + "?" + urlencode(params)

def domain_of(link: str) -> str:
    try:
        return urlparse(link).hostname or ""
    except Exception:
        return ""

def fetch_items_for_query(query: str, cap: int) -> list[dict]:
    url = google_news_rss_url(query)
    parsed = feedparser.parse(url)
    out = []
    for e in parsed.entries[: cap]:
        link = e.get("link") or ""
        dom = domain_of(link)
        if dom in BLOCKED_DOMAINS:
            continue
        out.append({
            "title": (e.get("title") or "").strip(),
            "link":  link,
            "published": (e.get("published") or e.get("updated") or "").strip(),
            "domain": dom,
        })
    return out

_analyzer = SentimentIntensityAnalyzer()

def label_sentiment(title: str) -> str:
    s = _analyzer.polarity_scores(title or "")
    v = s["compound"]
    return "positive" if v >= 0.2 else ("negative" if v <= -0.2 else "neutral")

def dedup_by_title_domain(items: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for x in items:
        key = (x["title"], x["domain"])
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out

def clean_for_theme(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokens_from(s: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z\-']+", (s or "").lower()))

def best_phrase_from(docs: list[str], stopwords: set[str], ngram_range=(2,3), min_df=2) -> str | None:
    """Return top phrase or None."""
    vec = CountVectorizer(stop_words=stopwords, ngram_range=ngram_range, min_df=min_df)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return None
    if X.shape[1] == 0:
        return None
    counts = X.sum(axis=0).A1
    vocab  = vec.get_feature_names_out()
    top_i  = counts.argmax()
    phrase = vocab[top_i]
    words  = phrase.split()
    if not words:
        return None
    if len(words) > MAX_THEME_WORDS:
        words = words[:MAX_THEME_WORDS]
    return " ".join(words)

def fallback_keywords(docs: list[str], stopwords: set[str], k: int = 3) -> str | None:
    """Return top-k unigrams as a phrase, or None."""
    vec = CountVectorizer(stop_words=stopwords, ngram_range=(1,1), min_df=1)
    try:
        X = vec.fit_transform(docs)
    except ValueError:
        return None
    if X.shape[1] == 0:
        return None
    counts = X.sum(axis=0).A1
    vocab  = vec.get_feature_names_out()
    pairs  = sorted(zip(counts, vocab), reverse=True)
    words  = [w for _, w in pairs[:k] if w]
    if not words:
        return None
    if len(words) > MAX_THEME_WORDS:
        words = words[:MAX_THEME_WORDS]
    return " ".join(words)

def theme_from_negatives(neg_titles: list[str], ceo: str, company: str) -> str:
    """
    Build a short theme from negative titles, excluding CEO/company tokens.
    Tries: (2–3)-grams min_df>=2, then bigrams min_df>=1, then top unigrams.
    """
    if not neg_titles:
        return "None"

    docs = [clean_for_theme(t) for t in neg_titles if t.strip()]
    if not docs:
        return "None"

    # dynamic stopwords (base + ceo/company tokens + 'ceo')
    stop = set(BASE_STOPWORDS)
    stop |= tokens_from(ceo)
    stop |= tokens_from(company)
    stop.add("ceo")

    # 1) strong signal: repeated 2–3 grams
    p = best_phrase_from(docs, stopwords=stop, ngram_range=(2,3), min_df=2)
    if p:
        return p

    # 2) any bigram
    p = best_phrase_from(docs, stopwords=stop, ngram_range=(2,2), min_df=1)
    if p:
        return p

    # 3) top unigrams (2–3 words)
    p = fallback_keywords(docs, stopwords=stop, k=3)
    return p if p else "None"

def ensure_dirs():
    os.makedirs(ARTICLES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

def purge_old_files():
    """Purge using date objects to avoid offset-naive/aware issues."""
    cutoff_date = date.today() - timedelta(days=PURGE_OLDER_THAN_DAYS)

    # delete old per-day article files
    for name in os.listdir(ARTICLES_DIR):
        if not name.endswith(".csv"):
            continue
        try:
            dt = datetime.strptime(name.replace(".csv", ""), "%Y-%m-%d").date()
        except Exception:
            continue
        if dt < cutoff_date:
            try:
                os.remove(os.path.join(ARTICLES_DIR, name))
            except Exception:
                pass

    # trim old rows in daily_counts.csv
    if os.path.exists(COUNTS_CSV):
        df = pd.read_csv(COUNTS_CSV)
        def keep(row):
            try:
                d = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
                return d >= cutoff_date
            except Exception:
                return True
        if not df.empty:
            df2 = df[df.apply(keep, axis=1)]
            if len(df2) != len(df):
                df2.to_csv(COUNTS_CSV, index=False)

# ------------------ Main ------------------

def main():
    print("=== CEO Sentiment (improved themes) : start ===")
    ensure_dirs()
    purge_old_files()

    ceo_to_company = load_ceo_company_map(ALIASES_CSV)
    if not ceo_to_company:
        raise SystemExit(f"No rows found in {ALIASES_CSV}. Expected header 'brand,alias' with alias=Company.")

    today = today_str()
    daily_articles_path = os.path.join(ARTICLES_DIR, f"{today}.csv")

    article_rows = []
    counts_rows  = []

    for idx, (ceo, company) in enumerate(ceo_to_company.items(), start=1):
        query = f"\"{ceo}\" \"{company}\""
        items = fetch_items_for_query(query, MAX_ITEMS_PER_QUERY)
        items = dedup_by_title_domain(items)

        pos = neu = neg = 0
        neg_titles = []

        for it in items:
            sent = label_sentiment(it["title"])
            if sent == "positive": pos += 1
            elif sent == "negative":
                neg += 1
                neg_titles.append(it["title"])
            else:
                neu += 1

            article_rows.append({
                "date": today,
                "brand": ceo,
                "company": company,
                "title": it["title"],
                "url": it["link"],
                "domain": it["domain"],
                "sentiment": sent,
                "published": it["published"],
            })

        total = pos + neu + neg
        theme = theme_from_negatives(neg_titles, ceo=ceo, company=company)

        counts_rows.append({
            "date": today,
            "brand": ceo,
            "company": company,
            "total": total,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "theme": theme,
        })

        if idx < len(ceo_to_company):
            time.sleep(REQUEST_PAUSE_SEC)

    # write per-article
    with open(daily_articles_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "date","brand","company","title","url","domain","sentiment","published"
        ])
        w.writeheader()
        for r in article_rows:
            w.writerow(r)

    # upsert counts
    if os.path.exists(COUNTS_CSV):
        df_old = pd.read_csv(COUNTS_CSV)
        df_old = df_old[df_old["date"] != today]
        df_new = pd.DataFrame(counts_rows, columns=[
            "date","brand","company","total","positive","neutral","negative","theme"
        ])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(COUNTS_CSV, index=False)
    else:
        with open(COUNTS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "date","brand","company","total","positive","neutral","negative","theme"
            ])
            w.writeheader()
            for r in counts_rows:
                w.writerow(r)

    print(f"Wrote {len(article_rows)} articles -> {daily_articles_path}")
    print(f"Upserted {len(counts_rows)} rows -> {COUNTS_CSV}")
    print("=== CEO Sentiment (improved themes) : done ===")

if __name__ == "__main__":
    main()
