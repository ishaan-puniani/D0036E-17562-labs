"""
Lab 2 — Linear and polynomial regression on Swedish income-by-age data.

Implements Tasks 1–5:
  - custom CSV loader (basic file I/O, no pandas/csv for loading)
  - from-scratch linear regression (vectorized / normal equation)
  - train/validation split, scatter plots, MSE
  - full-dataset cleaning + age grouping
  - sklearn polynomial regression with degree tuning
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
FIGURES_DIR = BASE_DIR / "figures"
SUBSET_PATH = BASE_DIR / "inc_subset.csv"
FULL_PATH = BASE_DIR / "inc_utf.csv"
RANDOM_STATE = 42
TRAIN_RATIO = 0.8


# ---------------------------------------------------------------------------
# Task 1 — Custom CSV loader (basic file I/O only)
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
        # Pad / trim so every row matches the header width
        if len(fields) < len(headers):
            fields.extend([""] * (len(headers) - len(fields)))
        elif len(fields) > len(headers):
            fields = fields[: len(headers)]
        rows.append(fields)
    return headers, rows


def rows_to_dataframe(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Optional helper: turn loader output into a DataFrame (allowed for saving / analysis)."""
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
    # Tip: set a breakpoint on the next line (or inspect `headers` / `rows` in the debugger)
    # to compare interactive inspection with this printout.
    _debug_probe = {"headers": headers, "nrows": len(rows), "sample": rows[:3]}
    print(f"Debugger probe dict keys: {list(_debug_probe.keys())}")


# ---------------------------------------------------------------------------
# Task 2 — From-scratch linear regression (vectorized / normal equation)
# ---------------------------------------------------------------------------


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Append a column of ones so θ0 (intercept) is learned with θ1..θn."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.c_[np.ones((X.shape[0], 1)), X]


def fit_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form (normal equation) linear regression:

        θ = (X_bᵀ X_b)⁻¹ X_bᵀ y

    where X_b includes a bias column. Matches the vectorized approach in
    week 3 / Hands-On ML chapter 4.
    """
    X_b = add_bias_column(X)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return theta.ravel()


def predict_linear(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    X_b = add_bias_column(X)
    return (X_b @ np.asarray(theta, dtype=float).reshape(-1, 1)).ravel()


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MSE = (1/n) Σ (y_i − ŷ_i)²  — implemented without sklearn."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_true - y_pred) ** 2))


def train_validation_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    random_state: int = RANDOM_STATE,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split into train / validation (default 80/20).

    For a cross-sectional age→income curve, a random shuffle is appropriate:
    a sequential split would put young ages only in train and older ages only
    in validation, which would not fairly test the fit across the age range.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(idx)
    cut = int(n * train_ratio)
    train_idx, val_idx = idx[:cut], idx[cut:]
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def plot_regression(
    X: np.ndarray,
    y: np.ndarray,
    theta: np.ndarray,
    title: str,
    output_path: Path,
    poly_model: Optional[object] = None,
    poly_label: Optional[str] = None,
) -> Path:
    """Scatter of all points + fitted line(s)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(X)
    Xs, ys = X[order], y[order]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(Xs, ys, alpha=0.75, label="data", edgecolors="none")

    x_line = np.linspace(Xs.min(), Xs.max(), 300)
    y_line = predict_linear(x_line, theta)
    ax.plot(x_line, y_line, color="C1", linewidth=2, label="linear regression")

    if poly_model is not None:
        y_poly = poly_model.predict(x_line.reshape(-1, 1))
        ax.plot(
            x_line,
            y_poly,
            color="C3",
            linewidth=2,
            linestyle="--",
            label=poly_label or "polynomial regression",
        )

    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Average income (2020)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def print_predictions(y_true: np.ndarray, y_pred: np.ndarray, label: str = "validation") -> None:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    print(f"\n{label.capitalize()} predictions (real vs predicted):")
    print(f"{'real':>12}  {'predicted':>12}  {'error':>12}")
    for real, pred in zip(y_true, y_pred):
        print(f"{real:12.4f}  {pred:12.4f}  {real - pred:12.4f}")


# ---------------------------------------------------------------------------
# Task 3 — Cleaning / grouping the full dataset
# ---------------------------------------------------------------------------


_AGE_RE = re.compile(r"(\d+)")


def parse_age(age_str: str) -> Optional[int]:
    """
    Convert age text like '20 years' or '100+ years' to an integer.
    Returns None if no digit is found.
    """
    match = _AGE_RE.search(str(age_str))
    return int(match.group(1)) if match else None


def subset_xy_from_loader(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    age_col: str = "age",
    income_col: str = "2020",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract numeric age / income arrays from the subset CSV loader output."""
    age_i = list(headers).index(age_col)
    inc_i = list(headers).index(income_col)
    ages, incomes = [], []
    for row in rows:
        try:
            ages.append(float(row[age_i]))
            incomes.append(float(row[inc_i]))
        except (ValueError, IndexError):
            continue
    return np.array(ages, dtype=float), np.array(incomes, dtype=float)


def full_grouped_xy(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    age_col: str = "age",
    income_col: str = "2020",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Clean age strings, convert income to float, and group by age across regions
    (mean income per age).
    """
    age_i = list(headers).index(age_col)
    inc_i = list(headers).index(income_col)

    records = []
    for row in rows:
        age = parse_age(row[age_i])
        if age is None:
            continue
        try:
            income = float(row[inc_i])
        except (ValueError, IndexError):
            continue
        records.append({"age": age, "income": income, "region": row[1] if len(row) > 1 else ""})

    df = pd.DataFrame(records)
    grouped = (
        df.groupby("age", as_index=False)["income"]
        .mean()
        .rename(columns={"income": "mean_income"})
        .sort_values("age")
        .reset_index(drop=True)
    )
    X = grouped["age"].to_numpy(dtype=float)
    y = grouped["mean_income"].to_numpy(dtype=float)
    return X, y, grouped


# ---------------------------------------------------------------------------
# Task 5 — Polynomial regression + hyperparameter tuning (sklearn)
# ---------------------------------------------------------------------------


class PolynomialRegressor:
    """
    PolynomialFeatures(degree) + LinearRegression.

    Ages are standardized before expanding powers so high degrees (e.g. 8)
    stay numerically stable (age^8 is huge on the raw 16–100 scale).
    """

    def __init__(self, degree: int):
        self.degree = degree
        self.pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
                ("lin", LinearRegression()),
            ]
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialRegressor":
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        self.pipeline.fit(X, np.asarray(y, dtype=float).ravel())
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float).reshape(-1, 1)
        return self.pipeline.predict(X)


def tune_polynomial_degree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    degrees: Iterable[int] = (2, 3, 5, 8),
) -> Tuple[int, dict, dict]:
    """Fit one model per degree; return (best_degree, mse_by_degree, models_by_degree)."""
    results = {}
    models: dict = {}
    best_degree = None
    best_mse = float("inf")

    for degree in degrees:
        model = PolynomialRegressor(degree).fit(X_train, y_train)
        mse = mean_squared_error(y_val, model.predict(X_val))
        results[degree] = mse
        models[degree] = model
        print(f"  degree={degree}: validation MSE = {mse:.4f}")
        if mse < best_mse:
            best_mse = mse
            best_degree = degree

    assert best_degree is not None
    return best_degree, results, models


def plot_all_polynomial_degrees(
    X: np.ndarray,
    y: np.ndarray,
    theta_linear: np.ndarray,
    models_by_degree: dict,
    mse_by_degree: dict,
    title_prefix: str,
    file_prefix: str,
) -> List[Path]:
    """
    Save one chart per polynomial degree (each vs the linear baseline),
    plus a multi-panel figure with all degrees side by side.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    degrees = list(models_by_degree.keys())

    for degree, model in models_by_degree.items():
        path = plot_regression(
            X,
            y,
            theta_linear,
            title=(
                f"{title_prefix} — degree={degree} "
                f"(val MSE={mse_by_degree[degree]:.2f})"
            ),
            output_path=FIGURES_DIR / f"{file_prefix}_degree_{degree}.png",
            poly_model=model,
            poly_label=f"polynomial degree={degree}",
        )
        print(f"Saved figure: {path}")
        saved.append(path)

    # Combined grid: all degrees at a glance
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    order = np.argsort(X)
    Xs, ys = X[order], y[order]
    x_line = np.linspace(Xs.min(), Xs.max(), 300)
    y_linear = predict_linear(x_line, theta_linear)

    n = len(degrees)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2 * nrows), sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, degree in zip(axes_flat, degrees):
        model = models_by_degree[degree]
        ax.scatter(Xs, ys, alpha=0.65, s=18, label="data", edgecolors="none")
        ax.plot(x_line, y_linear, color="C1", linewidth=1.8, label="linear")
        ax.plot(
            x_line,
            model.predict(x_line.reshape(-1, 1)),
            color="C3",
            linewidth=2,
            linestyle="--",
            label=f"poly deg={degree}",
        )
        ax.set_title(f"degree={degree}  |  val MSE={mse_by_degree[degree]:.2f}")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Average income (2020)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"{title_prefix} — all polynomial degrees", fontsize=13)
    fig.tight_layout()
    grid_path = FIGURES_DIR / f"{file_prefix}_all_degrees.png"
    fig.savefig(grid_path, dpi=140)
    plt.close(fig)
    print(f"Saved figure: {grid_path}")
    saved.append(grid_path)
    return saved


# ---------------------------------------------------------------------------
# Discussion helpers (printed with the run so answers are in one place)
# ---------------------------------------------------------------------------


def print_discussions(
    mse_subset: float,
    mse_full: float,
    poly_results: dict,
    best_degree: int,
) -> None:
    print("\n" + "=" * 72)
    print("DISCUSSION / REFLECTIONS")
    print("=" * 72)

    print(
        """
Task 1 — Debugger vs print
  • Debugger is better when you need to pause mid-execution, step line-by-line,
    and inspect several nested variables / shapes at a breakpoint (e.g. while
    debugging why a row failed to parse inside the loader loop).
  • Print is better for a quick, reproducible log of values across a full run
    (or when comparing many rows at once without interactive stepping), and
    it remains visible after the program finishes.

Task 2.1 — Train / validation split
  • We hold out ~20% so MSE reflects generalization, not just fit on the same
    points used to estimate θ. Without a hold-out, a model can look perfect
    while still being a poor predictor for unseen ages.
  • A random sample IS appropriate here: the data are a cross-section of ages
    in one year, not a time series. A sequential (age-ordered) split would put
    young ages in train and older ages in validation and unfairly stress the
    linear fit outside the trained age range.

Task 2.5 — Interpreting subset MSE
  • MSE is the average squared residual in (income units)². Smaller is better.
  • On the 20–50 subset the age→income relationship is close to linear and
    steadily increasing, so linear regression is a reasonable first model.
  • Remaining error comes from mild curvature (income growth slows at older
    ages in the subset) and from sampling variance in the validation split.
"""
    )
    print(f"  Subset validation MSE: {mse_subset:.4f}")
    print(f"  Full-data validation MSE: {mse_full:.4f}")

    print(
        f"""
Task 3.6 / 4.1 — Compare subset vs full-data linear regression
  • The full dataset spans ages 16–100+, and mean income rises then falls
    (career peak then retirement). A straight line cannot capture that
    non-monotonic shape, so residuals (and MSE) are much larger than on the
    nearly linear 20–50 subset.
  • Predictions on the full validation set look worse: young and very old ages
    are systematically mis-estimated by a single slope/intercept.

Task 4.2 — How to improve the full-data analysis
  • Use a non-linear model (polynomial / spline) or piecewise models
    (e.g. separate fits for working-age vs retirement ages).
  • Restrict the age range if the research question is about working-age income.
  • Add features (region, education proxies) instead of averaging all regions.
  • Regularize high-degree polynomials to avoid overfitting.

Task 5.3 — Best polynomial degree
  Validation MSEs by degree: { {d: round(m, 4) for d, m in poly_results.items()} }
  Best degree among the candidates: {best_degree}
  • We select the degree with the lowest validation MSE (not training MSE).
  • Degree 2 underfits the rise-then-fall curve. Higher degrees can follow the
    peak and retirement decline much more closely on this 1-D age curve.
  • Caveat: with a small validation set, a high degree can look excellent by
    chance; in practice you might still prefer a slightly worse but smoother
    degree (e.g. 3) or use regularization / more folds.
"""
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_linear_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    figure_name: str,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
    print(f"Samples: {len(y)}; age range: {X.min():.0f}–{X.max():.0f}")

    X_train, X_val, y_train, y_val = train_validation_split(X, y)
    print(f"Train size: {len(y_train)}; validation size: {len(y_val)}")

    theta = fit_linear_regression(X_train, y_train)
    print(f"Fitted θ (intercept, slope): {theta}")

    y_val_pred = predict_linear(X_val, theta)
    # Sort for readable side-by-side comparison by age
    order = np.argsort(X_val)
    print_predictions(y_val[order], y_val_pred[order], label="validation")

    mse = mean_squared_error(y_val, y_val_pred)
    print(f"\nValidation MSE: {mse:.4f}")

    out = plot_regression(
        X,
        y,
        theta,
        title=f"{name} — linear regression",
        output_path=FIGURES_DIR / figure_name,
    )
    print(f"Saved figure: {out}")
    return mse, theta, X_train, X_val, y_train, y_val


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ----- Task 1: load subset -----
    headers_sub, rows_sub = load_csv(SUBSET_PATH)
    print_loaded_preview(headers_sub, rows_sub, "Task 1: inc_subset.csv")
    # Allowed: save with pandas
    df_sub = rows_to_dataframe(headers_sub, rows_sub)
    df_sub.to_csv(BASE_DIR / "inc_subset_loaded.csv", index=False)
    print("Saved pandas copy: inc_subset_loaded.csv")

    X_sub, y_sub = subset_xy_from_loader(headers_sub, rows_sub)

    # ----- Task 2: linear regression on subset -----
    mse_sub, theta_sub, Xtr_s, Xva_s, ytr_s, yva_s = run_linear_experiment(
        "Task 2: Linear regression on income subset (ages 20–50)",
        X_sub,
        y_sub,
        "task2_subset_linear.png",
    )

    # ----- Task 3: load full dataset, clean, group -----
    headers_full, rows_full = load_csv(FULL_PATH)
    print_loaded_preview(headers_full, rows_full, "Task 3: inc_utf.csv", n=6)

    X_full, y_full, grouped = full_grouped_xy(headers_full, rows_full)
    print("\nTask 3.1 — Grouped by age (mean income across regions):")
    print(f"  Age groups: {len(grouped)} (vs {len(y_sub)} in the subset)")
    print(grouped.head(10).to_string(index=False))
    print("  ...")
    print(grouped.tail(5).to_string(index=False))
    grouped.to_csv(BASE_DIR / "inc_utf_grouped_by_age.csv", index=False)

    mse_full, theta_full, Xtr_f, Xva_f, ytr_f, yva_f = run_linear_experiment(
        "Task 3: Linear regression on full dataset (grouped by age)",
        X_full,
        y_full,
        "task3_full_linear.png",
    )

    # ----- Task 5: polynomial regression on the fuller / more curved data -----
    print(f"\n{'=' * 72}\nTask 5: Polynomial regression (sklearn) on full grouped data\n{'=' * 72}")
    degrees = (2, 3, 5, 8)
    print(f"Tuning polynomial degree over {degrees} ...")
    best_degree, poly_results, poly_models = tune_polynomial_degree(
        Xtr_f, ytr_f, Xva_f, yva_f, degrees=degrees
    )
    print(f"\nBest polynomial degree (lowest validation MSE): {best_degree}")
    print(
        f"Best poly validation MSE: {poly_results[best_degree]:.4f} "
        f"(linear was {mse_full:.4f})"
    )

    print("\nPlotting Task 5 charts for every degree (full data) ...")
    plot_all_polynomial_degrees(
        X_full,
        y_full,
        theta_full,
        poly_models,
        poly_results,
        title_prefix="Task 5: full data",
        file_prefix="task5_full",
    )
    # Keep a clear "best vs linear" figure for Task 5.4
    best_path = plot_regression(
        X_full,
        y_full,
        theta_full,
        title=f"Task 5.4: Linear vs best polynomial (degree={best_degree})",
        output_path=FIGURES_DIR / "task5_poly_vs_linear.png",
        poly_model=poly_models[best_degree],
        poly_label=f"polynomial degree={best_degree}",
    )
    print(f"Saved figure: {best_path}")

    # Also plot every degree on the subset for comparison
    print("\nPolynomial degree tuning on subset (for comparison):")
    best_d_sub, results_sub, models_sub = tune_polynomial_degree(
        Xtr_s, ytr_s, Xva_s, yva_s, degrees=degrees
    )
    print(f"Best subset polynomial degree: {best_d_sub} (MSE={results_sub[best_d_sub]:.4f})")
    print("\nPlotting Task 5 charts for every degree (subset) ...")
    plot_all_polynomial_degrees(
        X_sub,
        y_sub,
        theta_sub,
        models_sub,
        results_sub,
        title_prefix="Task 5: subset",
        file_prefix="task5_subset",
    )

    print_discussions(mse_sub, mse_full, poly_results, best_degree)
    print("\nDone. Figures are in:", FIGURES_DIR)


if __name__ == "__main__":
    main()
