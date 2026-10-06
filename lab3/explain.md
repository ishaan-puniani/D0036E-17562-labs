# Lab 3 — Discussion notes

Short answers for the K-means / silhouette lab. Numbers come from running `python3 lab3.py` (random seed 42).

## Task 1.1 — What the table shows

- One row per Swedish **county** for **2020**.
- Columns: annual rent per m² (SEK) and average yearly income (kSEK).
- Stockholm sits at the high-rent / high-income end; several northern/smaller counties are lower rent with moderate income.
- Useful for grouping regions with similar rent–income profiles for a real-estate campaign.

## Why no train / validation split?

K-means is **unsupervised**. There are no ground-truth labels to score a hold-out set against. All points help form the clusters; quality is judged with an **internal** metric (silhouette), not supervised validation MSE.

## Task 1.3 — K-means from scratch

- Initialize `k` centroids by sampling data points.
- Assign each point to the nearest centroid (Euclidean distance).
- Replace each centroid by its cluster mean; repeat ~10 times or until centroids stop moving.
- Colored scatter + centroid markers: see `figures/task1_kmeans_k3.png`.

## Task 2.1 — Silhouette grid search

- `a(i)` = mean distance to other points in the same cluster (intra).
- `b(i)` = mean distance to the nearest other cluster (inter).
- `S(i) = (b − a) / max(a, b)`; average over all points.
- `k = 1` is undefined (no other cluster) → skipped / NaN.
- With seed 42, **best k = 2** (mean silhouette ≈ 0.65). Next best among the grid is k=4 (~0.64), then k=3 (~0.63).

| k | mean silhouette |
|---|-----------------|
| 2 | 0.652 |
| 3 | 0.629 |
| 4 | 0.643 |
| 5–10 | lower / noisier |

Optimal `k=2` splits roughly into a **high-rent** cluster (Stockholm, Uppsala, Skåne) vs the remaining counties.

## Task 2.3 — New regions

| point | rent, income | assigned cluster |
|-------|--------------|------------------|
| new[0] | 1010, 320.12 | 1 (majority / lower-rent) |
| new[1] | 1258, 320 | 0 (high-rent trio) |
| new[2] | 980, 292.4 | 1 (majority / lower-rent) |

`new[1]` clearly matches the Stockholm/Uppsala/Skåne cloud. The other two have lower rent, so they land with the larger cluster even when income is relatively high (`new[0]`) — rent dominates Euclidean distance on the raw scale.

## Optional — N-D grid search

`nd_grid_search` takes a dict of hyperparameter lists (e.g. `n_clusters` × `max_iter`), evaluates every combination with `itertools.product`, and returns the best score/params.
