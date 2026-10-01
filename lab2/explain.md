# Lab 2 — Discussion notes

Short answers you can use with your lab partner. Numbers come from running `python3 lab2.py` (random seed 42).

## Task 1 — Debugger vs print

| Prefer | When |
|--------|------|
| **Debugger** | You need to pause inside a loop, step one line at a time, and inspect several variables/shapes at a breakpoint (e.g. a malformed CSV row). |
| **Print** | You want a lasting log of many values after the program ends, or a quick check without starting an interactive session. |

## Task 2.1 — Train / validation & random sample

- **Why split?** MSE on the training points alone can look optimistic. A hold-out (~20%) estimates how the model behaves on ages it did not use to fit θ.
- **Random sample?** Yes for this cross-sectional age→income data. A sequential split by sorted age would leave young ages in train and older ages only in validation, which unfairly tests extrapolation.

## Task 2.5 — Subset MSE & linear fit

- MSE = average squared residual in (income units)²; lower is better.
- Ages 20–50 show a mostly steady increase, so a straight line is a reasonable first model.
- Mild curvature (growth slowing toward 50) and split randomness explain remaining error.

## Tasks 3.6 / 4.1 — Subset vs full data

- Full data (16–100+) rises then falls; a single line cannot follow that shape → much higher MSE and worse validation predictions.
- Graphs: subset line tracks the cloud; full-data line is nearly flat/slightly declining through a curved cloud.

## Task 4.2 — Improvements

- Non-linear models (polynomial / spline) or piecewise fits (working age vs retirement).
- Narrow the age range if the question is about working-age income.
- Keep region as a feature instead of averaging all regions.
- Regularize high-degree polynomials.

## Task 5.3 — Best polynomial degree

- Evaluate degrees on the **validation** set (not training).
- After scaling ages, degree **8** has the lowest validation MSE among {2, 3, 5, 8} on the full grouped data; degree 2 underfits the rise-then-fall shape.
- Caveat: the validation set is small, so a high degree can look strong by chance — a smoother degree (e.g. 3) or cross-validation may be preferable in a real analysis.

Re-run `python3 lab2.py` to refresh exact MSE numbers after any code or data change.
