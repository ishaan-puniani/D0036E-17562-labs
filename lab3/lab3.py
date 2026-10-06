"""
Lab 3 — K-means clustering on Swedish rent vs income by region.

Implements Tasks 1–2 (+ optional N-D grid search):
  - custom CSV loader (same style as Lab 2)
  - from-scratch K-means (OOP)
  - silhouette score from scratch + grid search over k
  - classify new region datapoints
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
DATA_PATH = BASE_DIR / "inc_vs_rent.csv"
RANDOM_STATE = 42
KMEANS_MAX_ITER = 10

# New unnamed regions for Task 2.3: [annual rent sqm, avg yearly inc KSEK]
NEW_POINTS = np.array(
    [
        [1010.0, 320.12],
        [1258.0, 320.0],
        [980.0, 292.4],
    ],
    dtype=float,
)


# ---------------------------------------------------------------------------
# Task 1.1 — Custom CSV loader (basic file I/O only)
# ---------------------------------------------------------------------------


def load_csv(path: Path | str) -> Tuple[List[str], List[List[str]]]:
    """
    Load a CSV with basic file I/O (open / read / split).

    Does not use pandas or the csv module. Handles an optional UTF-8 BOM.
    Returns (headers, rows) where each row is a list of string fields.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig") as fh:
        text = fh.read()

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], []

    headers = [h.strip() for h in lines[0].split(",")]
    rows: List[List[str]] = []
    for line in lines[1:]:
        fields = [f.strip() for f in line.split(",")]
        if len(fields) < len(headers):
            fields.extend([""] * (len(headers) - len(fields)))
        elif len(fields) > len(headers):
            fields = fields[: len(headers)]
        rows.append(fields)
    return headers, rows


def rows_to_dataframe(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(headers))


def print_loaded_preview(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    title: str,
    n: int = 8,
) -> None:
    print(f"\n=== {title} ===")
    print(f"Headers ({len(headers)}): {headers}")
    print(f"Number of data rows: {len(rows)}")
    print(f"First {min(n, len(rows))} rows:")
    for i, row in enumerate(rows[:n]):
        print(f"  [{i}] {row}")
    if rows:
        print(f"Last row: {rows[-1]}")


def extract_xy(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix X = [rent, income] and region name list.

    Column names match inc_vs_rent.csv / rent_vs_inc.csv from Canvas.
    """
    # Skip leading empty / index column if present
    clean_headers = list(headers)
    if clean_headers and clean_headers[0] == "":
        name_to_idx = {h: i for i, h in enumerate(clean_headers)}
    else:
        name_to_idx = {h: i for i, h in enumerate(clean_headers)}

    rent_key = next(h for h in clean_headers if "rent" in h.lower())
    inc_key = next(h for h in clean_headers if "inc" in h.lower())
    region_key = next(h for h in clean_headers if "region" in h.lower())

    i_rent = name_to_idx[rent_key]
    i_inc = name_to_idx[inc_key]
    i_region = name_to_idx[region_key]

    X_list: List[List[float]] = []
    regions: List[str] = []
    for row in rows:
        rent = float(row[i_rent])
        income = float(row[i_inc])
        X_list.append([rent, income])
        regions.append(row[i_region])
    return np.asarray(X_list, dtype=float), regions


# ---------------------------------------------------------------------------
# Task 1.2 / 1.3 / 2.2 / 2.3 — Plotting helpers
# ---------------------------------------------------------------------------


def plot_scatter(
    X: np.ndarray,
    title: str,
    output_path: Path,
    labels: Optional[np.ndarray] = None,
    centroids: Optional[np.ndarray] = None,
    new_points: Optional[np.ndarray] = None,
    new_labels: Optional[np.ndarray] = None,
    region_names: Optional[Sequence[str]] = None,
) -> Path:
    """Scatter rent vs income; optionally color by cluster and mark new points."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 6.2))

    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=55, alpha=0.85, edgecolors="k", linewidths=0.4)
    else:
        labels = np.asarray(labels)
        n_clusters = int(labels.max()) + 1 if len(labels) else 0
        cmap = plt.get_cmap("tab10")
        for k in range(n_clusters):
            mask = labels == k
            ax.scatter(
                X[mask, 0],
                X[mask, 1],
                s=55,
                alpha=0.85,
                color=cmap(k % 10),
                edgecolors="k",
                linewidths=0.4,
                label=f"cluster {k}",
            )

    if centroids is not None:
        C = np.asarray(centroids, dtype=float)
        ax.scatter(
            C[:, 0],
            C[:, 1],
            s=180,
            marker="X",
            c="black",
            edgecolors="white",
            linewidths=1.0,
            label="centroids",
            zorder=5,
        )

    if new_points is not None:
        P = np.asarray(new_points, dtype=float)
        if new_labels is None:
            ax.scatter(
                P[:, 0],
                P[:, 1],
                s=120,
                marker="*",
                c="magenta",
                edgecolors="k",
                linewidths=0.6,
                label="new points",
                zorder=6,
            )
        else:
            cmap = plt.get_cmap("tab10")
            for i, (pt, lab) in enumerate(zip(P, np.asarray(new_labels))):
                ax.scatter(
                    pt[0],
                    pt[1],
                    s=140,
                    marker="*",
                    color=cmap(int(lab) % 10),
                    edgecolors="k",
                    linewidths=0.8,
                    label=f"new[{i}] → cluster {int(lab)}" if i < 3 else None,
                    zorder=6,
                )
                ax.annotate(
                    f"new[{i}]",
                    (pt[0], pt[1]),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=9,
                )

    ax.set_xlabel("Annual rent (SEK / m²)")
    ax.set_ylabel("Avg yearly income (kSEK)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    output_path = Path(output_path)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_silhouette_scores(scores: Dict[int, float], best_k: int, output_path: Path) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ks = sorted(scores.keys())
    vals = [scores[k] for k in ks]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ks, vals, marker="o", linewidth=2)
    ax.axvline(best_k, color="C3", linestyle="--", label=f"best k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Mean silhouette coefficient")
    ax.set_title("Task 2.1 — Silhouette grid search")
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Task 1.3 — K-means from scratch (OOP)
# ---------------------------------------------------------------------------


class KMeans:
    """
    From-scratch K-means clustering.

    Steps (Lab recap):
      1. Pick k
      2. Initialize centroids as random sample points
      3. Assign each point to nearest centroid (Euclidean)
      4. Recompute centroids as cluster means
      5. Repeat for max_iter (~10) or until centroids stop changing
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = KMEANS_MAX_ITER,
        random_state: int = RANDOM_STATE,
        tol: float = 1e-6,
    ):
        if n_clusters < 1:
            raise ValueError("n_clusters must be >= 1")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.random_state = random_state
        self.tol = float(tol)
        self.centroids_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.n_iter_: int = 0

    @staticmethod
    def euclidean_distances(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Return (n_samples, n_clusters) Euclidean distances."""
        X = np.asarray(X, dtype=float)
        C = np.asarray(centroids, dtype=float)
        # Broadcasting: (n, 1, d) - (1, k, d) → (n, k, d)
        return np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        k = min(self.n_clusters, n)
        idx = rng.choice(n, size=k, replace=False)
        return X[idx].copy()

    def _assign(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dists = self.euclidean_distances(X, centroids)
        return np.argmin(dists, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray, old: np.ndarray) -> np.ndarray:
        k = old.shape[0]
        new = old.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new[j] = members.mean(axis=0)
            # empty cluster: keep previous centroid
        return new

    def fit(self, X: np.ndarray) -> "KMeans":
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if self.n_clusters > len(X):
            raise ValueError("n_clusters cannot exceed number of samples")

        centroids = self._init_centroids(X)
        labels = self._assign(X, centroids)
        self.n_iter_ = 0

        for it in range(self.max_iter):
            self.n_iter_ = it + 1
            new_centroids = self._update_centroids(X, labels, centroids)
            shift = np.linalg.norm(new_centroids - centroids, axis=1).max()
            centroids = new_centroids
            labels = self._assign(X, centroids)
            if shift <= self.tol:
                break

        self.centroids_ = centroids
        self.labels_ = labels
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids_ is None:
            raise RuntimeError("Call fit() before predict().")
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.centroids_)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Task 2.1 — Silhouette coefficient from scratch + grid search
# ---------------------------------------------------------------------------


def pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    """Full (n, n) Euclidean distance matrix."""
    X = np.asarray(X, dtype=float)
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)


def intra_cluster_distance(i: int, labels: np.ndarray, dist: np.ndarray) -> float:
    """
    a(i): average distance from point i to other points in its own cluster.
    Returns 0.0 if the cluster has only one member.
    """
    same = np.where(labels == labels[i])[0]
    same = same[same != i]
    if len(same) == 0:
        return 0.0
    return float(dist[i, same].mean())


def nearest_inter_cluster_distance(i: int, labels: np.ndarray, dist: np.ndarray) -> float:
    """
    b(i): average distance from i to all points in the closest other cluster.
    """
    own = labels[i]
    other_clusters = [c for c in np.unique(labels) if c != own]
    if not other_clusters:
        return 0.0
    best = float("inf")
    for c in other_clusters:
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        avg = float(dist[i, members].mean())
        if avg < best:
            best = avg
    return best if best < float("inf") else 0.0


def silhouette_coefficient(i: int, labels: np.ndarray, dist: np.ndarray) -> float:
    """S(i) = (b(i) - a(i)) / max(a(i), b(i))."""
    a = intra_cluster_distance(i, labels, dist)
    b = nearest_inter_cluster_distance(i, labels, dist)
    denom = max(a, b)
    if denom == 0.0:
        return 0.0
    return (b - a) / denom


def mean_silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Average S(i) over all points.

    Undefined for a single cluster (k=1): return NaN so grid search can skip it.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    n_clusters = len(np.unique(labels))
    if n_clusters < 2 or len(X) < 2:
        return float("nan")

    dist = pairwise_euclidean(X)
    scores = [silhouette_coefficient(i, labels, dist) for i in range(len(X))]
    return float(np.mean(scores))


def silhouette_grid_search(
    X: np.ndarray,
    k_values: Iterable[int] = range(1, 11),
    max_iter: int = KMEANS_MAX_ITER,
    random_state: int = RANDOM_STATE,
) -> Tuple[int, Dict[int, float], Dict[int, KMeans]]:
    """
    For each k in the grid, fit KMeans and compute mean silhouette.

    Returns (best_k, scores_by_k, models_by_k). k=1 is recorded as NaN.
    """
    X = np.asarray(X, dtype=float)
    scores: Dict[int, float] = {}
    models: Dict[int, KMeans] = {}
    best_k = 2
    best_score = -float("inf")

    print("\nSilhouette grid search (k = 1..10):")
    for k in k_values:
        k = int(k)
        if k < 1 or k > len(X):
            scores[k] = float("nan")
            print(f"  k={k}: skipped (invalid for n={len(X)})")
            continue
        model = KMeans(n_clusters=k, max_iter=max_iter, random_state=random_state).fit(X)
        score = mean_silhouette_score(X, model.labels_)  # type: ignore[arg-type]
        scores[k] = score
        models[k] = model
        score_str = "nan (undefined for 1 cluster)" if np.isnan(score) else f"{score:.4f}"
        print(f"  k={k}: mean silhouette = {score_str}  (iters={model.n_iter_})")
        if not np.isnan(score) and score > best_score:
            best_score = score
            best_k = k

    return best_k, scores, models


# ---------------------------------------------------------------------------
# Optional advanced task — N-dimensional grid search
# ---------------------------------------------------------------------------


def nd_grid_search(
    param_grid: Dict[str, Sequence],
    score_fn: Callable[[dict], float],
    maximize: bool = True,
) -> Tuple[dict, float, List[Tuple[dict, float]]]:
    """
    Generic grid search over an arbitrary number of hyperparameters.

    param_grid example:
        {"n_clusters": range(2, 8), "max_iter": [5, 10, 20]}
    score_fn(params) -> float
    """
    keys = list(param_grid.keys())
    values = [list(param_grid[k]) for k in keys]
    results: List[Tuple[dict, float]] = []
    best_params: Optional[dict] = None
    best_score = -float("inf") if maximize else float("inf")

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        score = float(score_fn(params))
        results.append((params, score))
        better = score > best_score if maximize else score < best_score
        if not np.isnan(score) and better:
            best_score = score
            best_params = params

    if best_params is None:
        raise RuntimeError("Grid search produced no valid scores.")
    return best_params, best_score, results


# ---------------------------------------------------------------------------
# Discussion helpers
# ---------------------------------------------------------------------------


def print_discussions(
    regions: Sequence[str],
    X: np.ndarray,
    best_k: int,
    scores: Dict[int, float],
    labels: np.ndarray,
    new_labels: np.ndarray,
) -> None:
    print(f"\n{'=' * 72}")
    print("Discussion / interpretation notes")
    print("=" * 72)

    print(
        "\n1.1 Looking at the table:\n"
        "  - One row per Swedish county (region) for year 2020.\n"
        "  - Features: annual rent per m² (SEK) and average yearly income (kSEK).\n"
        "  - Stockholm has the highest rent; incomes also vary by region.\n"
        "  - Natural clustering question: group regions with similar rent–income profiles."
    )

    print(
        "\nWhy no train/validation split?\n"
        "  K-means is unsupervised: there is no held-out 'label' to predict.\n"
        "  Every point helps define the clusters. Quality is judged with an internal\n"
        "  metric (silhouette), not supervised validation MSE."
    )

    print(f"\n2.1 Optimal k by mean silhouette: {best_k}")
    for k in sorted(scores):
        v = scores[k]
        print(f"  k={k}: {v if np.isnan(v) else round(v, 4)}")

    print("\nCluster membership (optimal k):")
    for c in range(best_k):
        members = [regions[i] for i in range(len(regions)) if labels[i] == c]
        print(f"  cluster {c} ({len(members)}): {', '.join(members)}")

    print("\n2.3 New point assignments:")
    for i, (pt, lab) in enumerate(zip(NEW_POINTS, new_labels)):
        print(f"  new[{i}] rent={pt[0]:.1f}, income={pt[1]:.2f} → cluster {int(lab)}")
    print(
        "  Visually check the starred points against the colored clouds:\n"
        "  if they sit inside / nearest their assigned cluster, the prediction is reasonable."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Task 1.1: load dataset -----
    headers, rows = load_csv(DATA_PATH)
    print_loaded_preview(headers, rows, f"Task 1.1: {DATA_PATH.name}")
    df = rows_to_dataframe(headers, rows)
    df.to_csv(BASE_DIR / "inc_vs_rent_loaded.csv", index=False)
    print("Saved pandas copy: inc_vs_rent_loaded.csv")

    X, regions = extract_xy(headers, rows)
    print(f"\nFeature matrix shape: {X.shape}  (columns: rent, income)")
    print(f"Rent range: {X[:, 0].min():.0f}–{X[:, 0].max():.0f}")
    print(f"Income range: {X[:, 1].min():.2f}–{X[:, 1].max():.2f}")

    # ----- Task 1.2: scatter plot -----
    path_raw = plot_scatter(
        X,
        title="Task 1.2 — Regions: annual rent vs average income",
        output_path=FIGURES_DIR / "task1_scatter.png",
    )
    print(f"Saved figure: {path_raw}")

    # ----- Task 1.3: K-means from scratch (demo with k=3) -----
    print(f"\n{'=' * 72}\nTask 1.3: K-means from scratch (demo k=3)\n{'=' * 72}")
    demo_k = 3
    demo = KMeans(n_clusters=demo_k, max_iter=KMEANS_MAX_ITER, random_state=RANDOM_STATE).fit(X)
    print(f"Converged in {demo.n_iter_} iterations (max_iter={KMEANS_MAX_ITER})")
    print(f"Centroids (rent, income):\n{demo.centroids_}")
    path_demo = plot_scatter(
        X,
        title=f"Task 1.3 — K-means clusters (k={demo_k})",
        output_path=FIGURES_DIR / "task1_kmeans_k3.png",
        labels=demo.labels_,
        centroids=demo.centroids_,
    )
    print(f"Saved figure: {path_demo}")

    # ----- Task 2.1: silhouette grid search -----
    print(f"\n{'=' * 72}\nTask 2.1: Hyper-parameter optimization (silhouette)\n{'=' * 72}")
    best_k, scores, models = silhouette_grid_search(X, k_values=range(1, 11))
    print(f"\nBest k (highest mean silhouette): {best_k}  score={scores[best_k]:.4f}")
    path_sil = plot_silhouette_scores(
        {k: v for k, v in scores.items() if not np.isnan(v)},
        best_k,
        FIGURES_DIR / "task2_silhouette_grid.png",
    )
    print(f"Saved figure: {path_sil}")

    # ----- Task 2.2: scatter with optimal k -----
    best_model = models[best_k]
    path_best = plot_scatter(
        X,
        title=f"Task 2.2 — Optimal K-means (k={best_k})",
        output_path=FIGURES_DIR / f"task2_kmeans_k{best_k}.png",
        labels=best_model.labels_,
        centroids=best_model.centroids_,
    )
    print(f"Saved figure: {path_best}")

    # ----- Task 2.3: classify new points -----
    print(f"\n{'=' * 72}\nTask 2.3: Classify new unnamed regions\n{'=' * 72}")
    new_labels = best_model.predict(NEW_POINTS)
    for i, (pt, lab) in enumerate(zip(NEW_POINTS, new_labels)):
        print(f"  [{i}] rent={pt[0]}, income={pt[1]} → cluster {int(lab)}")
    path_new = plot_scatter(
        X,
        title=f"Task 2.3 — New points on optimal clusters (k={best_k})",
        output_path=FIGURES_DIR / "task2_new_points.png",
        labels=best_model.labels_,
        centroids=best_model.centroids_,
        new_points=NEW_POINTS,
        new_labels=new_labels,
    )
    print(f"Saved figure: {path_new}")

    # ----- Optional: N-dimensional grid search demo -----
    print(f"\n{'=' * 72}\nOptional: N-dimensional grid search\n{'=' * 72}")

    def score_params(params: dict) -> float:
        model = KMeans(
            n_clusters=int(params["n_clusters"]),
            max_iter=int(params["max_iter"]),
            random_state=RANDOM_STATE,
        ).fit(X)
        return mean_silhouette_score(X, model.labels_)  # type: ignore[arg-type]

    best_params, best_score, _ = nd_grid_search(
        {
            "n_clusters": range(2, 8),
            "max_iter": [5, 10, 20],
        },
        score_fn=score_params,
        maximize=True,
    )
    print(f"Best params: {best_params}  silhouette={best_score:.4f}")

    print_discussions(regions, X, best_k, scores, best_model.labels_, new_labels)  # type: ignore[arg-type]
    print("\nDone. Figures are in:", FIGURES_DIR)


if __name__ == "__main__":
    main()
