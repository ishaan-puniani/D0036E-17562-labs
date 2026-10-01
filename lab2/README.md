# Lab 2 — Linear & Polynomial Regression (Income by Age)

Custom CSV loading, from-scratch linear regression, and sklearn polynomial regression on Swedish average-income-by-age data (`inc_subset.csv`, `inc_utf.csv`).

## Prerequisites

- Python 3.9+
- pip

## Setup

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 lab2.py
```

This prints:

1. Loaded CSV previews (Task 1 & 3) — also useful to inspect with the debugger  
2. Train/validation linear regression on the subset and full grouped data  
3. Real vs predicted validation values and MSE  
4. Polynomial degree tuning (2, 3, 5, 8) and discussion answers  

Figures are written to `figures/`.

## Project layout

| File / folder | Description |
|---------------|-------------|
| `lab2.py` | Full Tasks 1–5 implementation |
| `inc_subset.csv` | Ages 20–50 average income (2020) |
| `inc_utf.csv` | Income by age and region (2020) |
| `requirements.txt` | Python dependencies |
| `figures/` | Scatter + regression plots |

## Notes

- **CSV loading** uses only basic file I/O (`open` / `read` / `split`). Pandas is used afterward to save cleaned tables.
- **Linear regression** uses the vectorized normal equation (no sklearn).
- **MSE** is implemented manually.
- **Polynomial regression** uses `PolynomialFeatures` + `LinearRegression` from scikit-learn.
- Set a breakpoint in `print_loaded_preview` (or on `_debug_probe`) to satisfy the debugger inspection requirement in Tasks 1 and 3.
