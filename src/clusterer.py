"""
clusterer.py — Embedding + Semantic Clustering Layer
Embeds reviews with sentence-transformers, reduces dimensionality, and
clusters with HDBSCAN (silhouette-selected KMeans fallback) — the number
of clusters is discovered from the data, not predetermined. Saves labeled
cluster assignments to data/clusters.json and data/clusters_reviews.csv.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────

def embed_reviews(df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed review texts using a local sentence-transformer model.
    Falls back to TF-IDF if sentence-transformers isn't available.
    """
    texts = (df["title"].fillna("") + " " + df["text"]).tolist()

    try:
        from sentence_transformers import SentenceTransformer
        log.info(f"Embedding {len(texts)} reviews with {model_name}...")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
        log.info(f"  → Embeddings shape: {embeddings.shape}")
        return embeddings

    except ImportError:
        log.warning("sentence-transformers not available. Falling back to TF-IDF.")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        vectorizer = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(50, tfidf_matrix.shape[1] - 1))
        embeddings = svd.fit_transform(tfidf_matrix)
        log.info(f"  → TF-IDF SVD embeddings shape: {embeddings.shape}")
        return embeddings


# ─────────────────────────────────────────────
# Dimensionality Reduction
# ─────────────────────────────────────────────

def reduce_dimensions(embeddings: np.ndarray, n_components: int = 10) -> np.ndarray:
    """Reduce to lower dims for clustering. Uses UMAP if available, else PCA."""
    try:
        import umap
        log.info("Reducing dimensions with UMAP...")
        reducer = umap.UMAP(n_components=n_components, random_state=42, min_dist=0.1, n_neighbors=15)
        reduced = reducer.fit_transform(embeddings)
        return reduced
    except ImportError:
        log.warning("UMAP not available. Using PCA instead.")
        from sklearn.decomposition import PCA
        n = min(n_components, embeddings.shape[1], embeddings.shape[0] - 1)
        pca = PCA(n_components=n, random_state=42)
        return pca.fit_transform(embeddings)


def reduce_to_2d(embeddings: np.ndarray) -> np.ndarray:
    """Reduce to 2D for visualization."""
    try:
        import umap
        reducer = umap.UMAP(n_components=2, random_state=42, min_dist=0.2, n_neighbors=10)
        return reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings)


# ─────────────────────────────────────────────
# Clustering
# ─────────────────────────────────────────────
# Cluster count is DISCOVERED from the data, not predetermined. The
# original implementation forced KMeans to exactly k=5 ("to match PM
# framework") — that inverts the scientific method: it decides how many
# distinct failure modes exist before looking at the data. HDBSCAN is
# density-based and finds however many natural clusters the data
# supports (it might be 3, it might be 7). KMeans is only used as a
# fallback when hdbscan isn't installed, and even then the k is chosen
# by silhouette score across a range — never hardcoded.

def cluster_reviews(
    reduced_embeddings: np.ndarray,
    min_cluster_size: int = 15,
) -> np.ndarray:
    """
    Cluster embeddings, letting the natural number of clusters emerge.
    Tries HDBSCAN first (no k required). Falls back to KMeans with a
    silhouette-selected k only if hdbscan isn't installed.
    """
    try:
        import hdbscan
        log.info(f"Clustering with HDBSCAN (min_cluster_size={min_cluster_size})...")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(reduced_embeddings)
        n_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        log.info(f"  → HDBSCAN found {n_found} natural clusters ({n_noise} noise points, label -1)")
        if n_found == 0:
            log.warning("  HDBSCAN found 0 clusters at this min_cluster_size. Falling back to KMeans.")
            raise ValueError("HDBSCAN found no clusters")
        return labels

    except (ImportError, ValueError) as e:
        log.warning(f"Using KMeans with silhouette-selected k ({e if isinstance(e, ValueError) else 'hdbscan not installed'}).")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n = len(reduced_embeddings)
        k_max = min(9, n - 1)
        if k_max < 3:
            log.warning("Too few reviews to cluster meaningfully. Returning a single cluster.")
            return np.zeros(n, dtype=int)

        best_k, best_score, best_labels = None, -1.0, None
        for k in range(3, k_max + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(reduced_embeddings)
            try:
                score = silhouette_score(reduced_embeddings, labels)
            except ValueError:
                continue
            log.info(f"  KMeans k={k}: silhouette={score:.3f}")
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        log.info(
            f"  → KMeans selected k={best_k} by silhouette score ({best_score:.3f}) — "
            f"chosen from the data, not a predetermined target"
        )
        return best_labels


# ─────────────────────────────────────────────
# Consolidate genuinely small clusters
# ─────────────────────────────────────────────

def consolidate_small_clusters(
    df: pd.DataFrame,
    raw_labels: np.ndarray,
    embeddings: np.ndarray,
    min_cluster_pct: float = 3.0,
    max_clusters: int = 8,
) -> pd.DataFrame:
    """
    Merge clusters too small to be a trustworthy, actionable failure mode
    (below min_cluster_pct of total reviews) into their nearest larger
    cluster by centroid distance.

    This is deliberately NOT "force down to N clusters." Clusters are
    only merged if they're statistically too small to stand on their
    own; the survivor count is whatever the data supports. max_clusters
    is a presentation backstop only — it keeps a dashboard/deck from
    having to show a long tail of micro-clusters, and merges the
    smallest survivors first if (and only if) more than max_clusters
    remain after the min_cluster_pct pass.

    Returns df with 'cluster_id' (final, contiguous 0..N-1, ordered by
    size descending) and 'cluster_raw' (original label) columns.
    """
    df = df.copy()
    df["cluster_raw"] = raw_labels

    noise_mask = raw_labels == -1
    valid_labels = raw_labels[~noise_mask]

    if len(valid_labels) == 0:
        log.warning("No valid clusters found (all noise). Returning a single cluster.")
        df["cluster_id"] = 0
        return df

    cluster_sizes = pd.Series(valid_labels).value_counts()
    total_valid = len(valid_labels)
    log.info(f"Raw cluster sizes (pre-consolidation): {cluster_sizes.to_dict()}")

    centroids = {
        lbl: embeddings[raw_labels == lbl].mean(axis=0)
        for lbl in cluster_sizes.index
    }

    small_labels = [
        lbl for lbl in cluster_sizes.index
        if (cluster_sizes[lbl] / total_valid * 100) < min_cluster_pct
    ]
    large_labels = [lbl for lbl in cluster_sizes.index if lbl not in small_labels]

    if not large_labels:
        # Every cluster is "small" relative to total volume — keep them
        # all rather than collapsing genuinely distinct signal into one
        # meaningless bucket.
        large_labels = list(cluster_sizes.index)
        small_labels = []

    label_map = {lbl: lbl for lbl in large_labels}
    for lbl in small_labels:
        dists = {l2: float(np.linalg.norm(centroids[lbl] - centroids[l2])) for l2 in large_labels}
        nearest = min(dists, key=dists.get)
        label_map[lbl] = nearest
        log.info(
            f"  Merging small cluster {lbl} ({cluster_sizes[lbl]} reviews, "
            f"{cluster_sizes[lbl] / total_valid * 100:.1f}% < {min_cluster_pct}% threshold) "
            f"into cluster {nearest}"
        )

    # Presentation backstop: if more than max_clusters survive the
    # min_cluster_pct pass, merge the smallest survivors into their
    # nearest remaining neighbor until the count is at or below the cap.
    surviving = sorted(set(label_map.values()))
    while len(surviving) > max_clusters:
        sizes_now = {
            lbl: sum(cluster_sizes.get(orig, 0) for orig, mapped in label_map.items() if mapped == lbl)
            for lbl in surviving
        }
        smallest = min(sizes_now, key=sizes_now.get)
        remaining = [l for l in surviving if l != smallest]
        dists = {l2: float(np.linalg.norm(centroids[smallest] - centroids[l2])) for l2 in remaining}
        nearest = min(dists, key=dists.get)
        for orig, mapped in list(label_map.items()):
            if mapped == smallest:
                label_map[orig] = nearest
        log.info(
            f"  Presentation cap: merging cluster {smallest} into {nearest} "
            f"(more than {max_clusters} clusters survived the size threshold)"
        )
        surviving = sorted(set(label_map.values()))

    label_map[-1] = -1  # noise reassigned separately, below
    df["cluster_id"] = df["cluster_raw"].map(label_map)

    # Reassign noise points to their nearest surviving cluster by centroid
    # distance (more defensible than taking the mode of the whole dataset,
    # which is what the original implementation did).
    if noise_mask.sum() > 0:
        surviving_final = sorted(set(df.loc[~noise_mask, "cluster_id"].unique()))
        noise_embeddings = embeddings[noise_mask]
        reassigned = []
        for emb in noise_embeddings:
            dists = {c: float(np.linalg.norm(emb - centroids[c])) for c in surviving_final}
            reassigned.append(min(dists, key=dists.get))
        df.loc[noise_mask, "cluster_id"] = reassigned

    # Final relabel: contiguous 0..N-1, ordered by size descending, so a
    # PM deck/dashboard never has to explain gaps like "cluster 0, 2, 5".
    final_sizes = df["cluster_id"].value_counts()
    relabel = {old: new for new, old in enumerate(final_sizes.index)}
    df["cluster_id"] = df["cluster_id"].map(relabel)

    n_final = df["cluster_id"].nunique()
    log.info(f"✓ Consolidation complete: {n_final} final clusters (emerged from data)")

    return df


# ─────────────────────────────────────────────
# Extract Representative Quotes
# ─────────────────────────────────────────────

def get_representative_quotes(cluster_df: pd.DataFrame, n: int = 5) -> list[str]:
    """
    Get the most representative quotes from a cluster.
    Prioritizes: longer reviews, higher helpful_votes, diversity of source.
    """
    scored = cluster_df.copy()
    scored["score"] = (
        scored["word_count"].fillna(0).clip(0, 200) / 200 * 0.4
        + scored["helpful_votes"].fillna(0).clip(0, 500) / 500 * 0.6
    )
    scored = scored.sort_values("score", ascending=False)

    quotes = []
    seen_sources = set()
    for _, row in scored.iterrows():
        if len(quotes) >= n:
            break
        # Trim to a clean excerpt (first 280 chars)
        text = row["text"].strip()
        excerpt = text[:280] + ("..." if len(text) > 280 else "")
        quotes.append({
            "text": excerpt,
            "source": row["source"],
            "provenance": row.get("provenance", "scraped"),
            "rating": row.get("rating"),
            "helpful_votes": int(row.get("helpful_votes", 0)),
        })
        seen_sources.add(row["source"])

    return quotes


# ─────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────

def run_clustering_pipeline(
    df: pd.DataFrame,
    output_path: str = "data/clusters.json",
    min_cluster_size: int = 15,
    min_cluster_pct: float = 3.0,
    max_clusters: int = 8,
) -> dict:
    """
    Full clustering pipeline: embed → reduce dimensions → cluster (natural
    count) → consolidate small clusters → 2D layout for visualization →
    cluster metadata. Returns (clusters_meta, df_clustered).
    """
    log.info(f"Starting clustering pipeline on {len(df)} reviews...")

    # 1. Embed
    embeddings = embed_reviews(df)

    # 2. Reduce dimensions BEFORE clustering. High-dimensional embeddings
    # (384-d for MiniLM) degrade Euclidean-distance-based clustering —
    # this step exists in the file but was never actually called by the
    # pipeline before this fix.
    log.info("Reducing dimensions before clustering...")
    reduced = reduce_dimensions(embeddings, n_components=10)

    # 3. Cluster — natural count, not forced
    raw_labels = cluster_reviews(reduced, min_cluster_size=min_cluster_size)

    # 4. Consolidate only clusters too small to be a trustworthy failure
    # mode; cap survivor count at max_clusters as a presentation backstop.
    df_clustered = consolidate_small_clusters(
        df, raw_labels, reduced,
        min_cluster_pct=min_cluster_pct,
        max_clusters=max_clusters,
    )

    # 5. 2D reduction for visualization. Column names are umap_x/umap_y —
    # this is what app.py's render_cluster_scatter() actually reads. The
    # previous version wrote 'x'/'y' here, which is the exact reason the
    # semantic cluster map never rendered.
    log.info("Computing 2D layout for visualization...")
    coords_2d = reduce_to_2d(embeddings)
    df_clustered["umap_x"] = coords_2d[:, 0]
    df_clustered["umap_y"] = coords_2d[:, 1]

    # 6. Build cluster summaries
    clusters_meta = {}
    for cluster_id in sorted(df_clustered["cluster_id"].unique()):
        cluster_df = df_clustered[df_clustered["cluster_id"] == cluster_id]
        quotes = get_representative_quotes(cluster_df, n=6)
        avg_rating = cluster_df["rating"].dropna().mean() if "rating" in cluster_df else None

        clusters_meta[int(cluster_id)] = {
            "cluster_id": int(cluster_id),
            "size": len(cluster_df),
            "pct_of_total": round(len(cluster_df) / len(df) * 100, 1),
            "avg_rating": round(avg_rating, 2) if avg_rating and not np.isnan(avg_rating) else None,
            "representative_quotes": quotes,
            "sources": cluster_df["source"].value_counts().to_dict() if "source" in cluster_df else {},
        }

    log.info(
        f"✓ {len(clusters_meta)} clusters identified from the data "
        f"(min_cluster_size={min_cluster_size}, min_cluster_pct={min_cluster_pct}%, "
        f"max_clusters cap={max_clusters})"
    )

    # 7. Save clustered CSV
    csv_path = output_path.replace(".json", "_reviews.csv")
    df_clustered.to_csv(csv_path, index=False)
    log.info(f"✓ Saved clustered reviews → {csv_path}")

    # 8. Save cluster metadata JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clusters_meta, f, indent=2)
    log.info(f"✓ Saved cluster metadata → {output_path}")

    return clusters_meta, df_clustered


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv("data/raw_reviews.csv")
    meta, df_c = run_clustering_pipeline(df)
    for cid, info in meta.items():
        print(f"\nCluster {cid}: {info['size']} reviews ({info['pct_of_total']}%)")
        print(f"  Sample: {info['representative_quotes'][0]['text'][:120]}...")
