# Understanding `explore_housing.py` (for Node.js developers)

This note explains how the housing exploration script works, especially where
columns like `df["median_house_value"]` come from, and what the common pandas
methods mean in Node.js terms.

## 1. Where does `df["median_house_value"]` come from?

### It starts in the CSV header

The first line of `housing.csv` looks like:

```text
longitude,latitude,...,median_income,median_house_value,ocean_proximity
```

`median_house_value` is a **column name** in the file — like a property name on
each row object in JavaScript.

### `pd.read_csv` builds a table (`DataFrame`)

```python
df = pd.read_csv(path)
```

Rough Node equivalent:

```js
// pseudo-Node
const rows = await csvParse(fs.readFileSync("housing.csv"));
// rows ≈ [
//   { longitude: -122.23, ..., median_house_value: 452600, ocean_proximity: "NEAR BAY" },
//   { longitude: -122.22, ..., median_house_value: 358500, ocean_proximity: "NEAR BAY" },
// ]
```

In pandas that table is `df` (a DataFrame):
- **rows** = districts / geographical units
- **columns** = fields from the CSV header

### `df["median_house_value"]` selects one column

```python
df["median_house_value"]
```

Node analogy:

```js
rows.map((r) => r.median_house_value);
// [452600, 358500, 352100, ...]
```

In pandas that column is a **Series** (a typed 1D array with an index).

You can also write `df.median_house_value` — similar to `obj.prop` vs
`obj["prop"]` in JS. Bracket form is safer for odd names.

### Then you call methods on that column

```python
float(df["median_house_value"].mean())
```

Flow:
1. `df` → whole table
2. `df["median_house_value"]` → just that column
3. `.mean()` → average of those numbers

Node-ish:

```js
const values = rows.map((r) => r.median_house_value);
const mean = values.reduce((a, b) => a + b, 0) / values.length;
```

### Where `df` comes from in this project

In `main()`:

```python
df = load_housing()  # reads CSV, returns DataFrame
```

`load_housing()`:
1. reads `housing.csv` → `df`
2. **adds** new columns (not from the CSV), for example:

```python
df["rooms_per_household"] = df["total_rooms"] / df["households"]
```

Node analogy:

```js
rows = rows.map((r) => ({
  ...r,
  rooms_per_household: r.total_rooms / r.households,
}));
```

So:
- `df["median_house_value"]` → came from the CSV column
- `df["rooms_per_household"]` → computed in Python after load

### Tiny mental model

| Python / pandas | Node mental model |
|-----------------|-------------------|
| `DataFrame` (`df`) | array of objects / table |
| column name | object key |
| `df["median_house_value"]` | `rows.map(r => r.median_house_value)` |
| `Series` | typed array of one field |
| `.mean()` | average of that array |

**Bottom line:** `df["median_house_value"]` does not invent a name — it looks up
the column pandas created from the `median_house_value` header when you ran
`pd.read_csv(...)`.

---

## 2. Other methods used in `explore_housing.py`

### Load & table basics

**`pd.read_csv(path)`**  
Reads CSV → DataFrame.  
≈ `csv-parse` / parsing a file into `[{...}, {...}]`.

**`len(df)`**  
Number of rows.  
≈ `rows.length`.

**`df["a"] / df["b"]`**  
Element-wise divide two columns → new Series.  
≈ `rows.map(r => r.a / r.b)`.

**`df["new_col"] = ...`**  
Adds/overwrites a column.  
≈ mapping and attaching a new property on each object.

**`float(...)` / `int(...)`**  
Cast to Python number types.  
≈ `Number(...)` / `Math.trunc(...)`.

### Aggregations on one column

- **`.mean()`** — average  
- **`.median()`** — middle value  
- **`.max()`** — largest value  
- **`.sum()`** — add values (also used to count `True`s)

```js
// .mean()
values.reduce((a, b) => a + b, 0) / values.length;

// .max()
Math.max(...values);
```

### Grouping (SQL / lodash core)

**`df.groupby("ocean_proximity")`**  
Split rows by category.  
≈ `_.groupBy(rows, "ocean_proximity")` or SQL `GROUP BY`.

```python
df.groupby("ocean_proximity")["median_house_value"].mean()
```

≈:

```js
Object.fromEntries(
  Object.entries(_.groupBy(rows, "ocean_proximity")).map(([k, arr]) => [
    k,
    average(arr.map((r) => r.median_house_value)),
  ])
);
```

**`.agg(...)`** — run several stats at once.

```python
# named outputs from one column
.agg(mean="mean", median="median", count="count")

# named outputs from different columns
.agg(
  mean_house_value=("median_house_value", "mean"),
  n=("median_house_value", "size"),
)
```

- `"size"` ≈ group length (`arr.length`)
- `"count"` ≈ non-null count

**`.sort_values("mean", ascending=False)`**  
Sort by a column.  
≈ `arr.sort((a, b) => b.mean - a.mean)`.

**`.round(2)`**  
Round numbers for display.

### Filtering / counting / missing data

**`df["age"] == age_max`**  
Boolean Series per row.  
≈ `rows.map(r => r.age === ageMax)`.

**`(df["age"] == age_max).sum()`**  
Count of `True`.  
≈ `rows.filter(r => r.age === ageMax).length`.

**`.isna()`**  
Where values are missing.  
≈ null checks.

**`.isna().sum()`**  
Count missing values.

**`.dropna()`**  
Drop missing values before plotting.  
≈ `values.filter(v => v != null)`.

### Selecting multiple columns & uniqueness

**`df[["longitude", "latitude"]]`**  
Pick several columns.  
≈ `rows.map(r => ({ longitude: r.longitude, latitude: r.latitude }))`.

**`.drop_duplicates()`**  
Unique rows.

**`.shape`**  
`(rows, cols)` tuple → `.shape[0]` is row count.  
≈ `rows.length`.

### Category counts & correlation

**`.value_counts()`**  
Frequency of each category (like a frequency map / reduce counter).

**`.to_dict()`**  
Series/DataFrame → plain Python dict.  
≈ turning a Map into a plain object.

**`.corr()`**  
Correlation matrix between numeric columns (−1…1).  
Example: `0.688` between income and house value = strong positive link.

**`.loc["median_income", "median_house_value"]`**  
Label-based lookup.  
≈ `corrMatrix["median_income"]["median_house_value"]`.

### Series helpers when printing

**`.apply(lambda x: f"${x:,.2f}")`**  
Map a function over each value.  
≈ `series.map(x => formatMoney(x))`.  
`lambda` is an inline arrow function: `x => ...`.

**f-strings** — `f"mean: ${overall_mean:,.2f}"`  
≈ JS template literals with formatting.

### Matplotlib (plotting), briefly

| Call | What it does |
|------|----------------|
| `plt.subplots(2, 2)` | Create a 2×2 grid of charts → `(fig, axes)` |
| `axes.ravel()` | Flatten 2D axes array → 4 axes |
| `zip(axes, columns)` | Pair each axis with a column name |
| `ax.hist(...)` | Draw histogram |
| `ax.set_title` / `xlabel` / `ylabel` | Labels |
| `fig.savefig(path)` | Write PNG |
| `plt.close(fig)` | Free memory |

**`Path` / `.mkdir` / `path / "file.png"`**  
Pathlib ≈ Node `path` + bits of `fs`.

### Method chaining

```python
df.groupby("ocean_proximity")["median_house_value"].mean().sort_values(ascending=False)
```

Left → right, like a lodash chain: group → average → sort descending.

---

## 3. Quick cheat sheet

| pandas | Node mental model |
|--------|-------------------|
| `read_csv` | parse CSV → array of objects |
| `df["col"]` | `map(r => r.col)` |
| `groupby` | `groupBy` / SQL `GROUP BY` |
| `mean` / `median` / `max` / `sum` | math on an array |
| `agg` | multiple reductions per group |
| `sort_values` | `sort` |
| `isna` / `dropna` | null checks / filter |
| `value_counts` | frequency map |
| `corr` | correlation matrix |
| `apply` | `map` |
| `loc` | keyed object access |

## 4. How to run the project

```bash
python3 -m pip install -r requirements.txt
python3 explore_housing.py
```

- Answers for Q1–Q10 print in the terminal
- Histograms are saved to `figures/histograms.png`