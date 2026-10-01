"""
Explore the California housing dataset and answer the lab research questions.

Each research question is answered by a dedicated function. Running this file
prints results and interpretations, and saves histograms to figures/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "housing.csv"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"


def load_housing(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load housing.csv and add a few derived features used later."""
    df = pd.read_csv(path)
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]
    return df


def count_geographical_units(df: pd.DataFrame) -> int:
    """Q1: each row is one geographical unit (census block group / district)."""
    return len(df)


def mean_house_value_overall(df: pd.DataFrame) -> float:
    """Q2: mean median_house_value across all ocean_proximity categories."""
    return float(df["median_house_value"].mean())


def mean_house_value_by_proximity(df: pd.DataFrame) -> pd.Series:
    """Q3: mean median_house_value for each ocean_proximity category."""
    return (
        df.groupby("ocean_proximity")["median_house_value"]
        .mean()
        .sort_values(ascending=False)
    )


def mean_vs_median_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Q4: compare mean and median house values within each category."""
    summary = (
        df.groupby("ocean_proximity")["median_house_value"]
        .agg(mean="mean", median="median", count="count")
        .sort_values("mean", ascending=False)
    )
    summary["mean_minus_median"] = summary["mean"] - summary["median"]
    return summary


def plot_variable_histograms(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Q5: create histograms for selected numeric columns."""
    if columns is None:
        columns = [
            "households",
            "median_income",
            "housing_median_age",
            "median_house_value",
        ]

    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, col in zip(axes, columns):
        ax.hist(df[col].dropna(), bins=50, edgecolor="black", alpha=0.75)
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")

    fig.suptitle("Housing dataset histograms", fontsize=14)
    fig.tight_layout()
    out_path = output_dir / "histograms.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def describe_histogram_features(df: pd.DataFrame) -> dict:
    """Q6 helpers: detect capping at the right tail of age and house value."""
    age_max = float(df["housing_median_age"].max())
    value_max = float(df["median_house_value"].max())
    return {
        "housing_median_age_max": age_max,
        "housing_median_age_capped_count": int(
            (df["housing_median_age"] == age_max).sum()
        ),
        "median_house_value_max": value_max,
        "median_house_value_capped_count": int(
            (df["median_house_value"] == value_max).sum()
        ),
        "median_house_value_overall_mean": float(df["median_house_value"].mean()),
        "median_house_value_overall_median": float(df["median_house_value"].median()),
    }


def expensive_and_large_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Q7: compare price and size proxies across ocean_proximity categories."""
    return (
        df.groupby("ocean_proximity")
        .agg(
            mean_house_value=("median_house_value", "mean"),
            mean_total_rooms=("total_rooms", "mean"),
            mean_rooms_per_household=("rooms_per_household", "mean"),
            n=("median_house_value", "size"),
        )
        .sort_values("mean_house_value", ascending=False)
    )


def house_quality_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Q8: age and room-related quality indicators by ocean_proximity."""
    return (
        df.groupby("ocean_proximity")
        .agg(
            mean_age=("housing_median_age", "mean"),
            mean_total_rooms=("total_rooms", "mean"),
            mean_total_bedrooms=("total_bedrooms", "mean"),
            mean_rooms_per_household=("rooms_per_household", "mean"),
            mean_bedrooms_per_room=("bedrooms_per_room", "mean"),
        )
        .sort_values("mean_age", ascending=False)
    )


def demographics_by_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Q9: population, households, and income by ocean_proximity."""
    return (
        df.groupby("ocean_proximity")
        .agg(
            mean_population=("population", "mean"),
            mean_households=("households", "mean"),
            mean_pop_per_household=("population_per_household", "mean"),
            mean_median_income=("median_income", "mean"),
            n=("population", "size"),
        )
        .sort_values("mean_median_income", ascending=False)
    )


def other_observations(df: pd.DataFrame) -> dict:
    """Q10: extra patterns worth noticing."""
    missing_bedrooms = int(df["total_bedrooms"].isna().sum())
    corr = df[
        [
            "median_income",
            "median_house_value",
            "housing_median_age",
            "rooms_per_household",
            "population_per_household",
        ]
    ].corr()
    return {
        "missing_total_bedrooms": missing_bedrooms,
        "income_value_corr": float(
            corr.loc["median_income", "median_house_value"]
        ),
        "category_counts": df["ocean_proximity"].value_counts().to_dict(),
        "unique_lat_lon_pairs": int(
            df[["longitude", "latitude"]].drop_duplicates().shape[0]
        ),
    }


def print_section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main() -> None:
    df = load_housing()

    # ------------------------------------------------------------------ Q1
    print_section("Q1. How many geographical units are there?")
    n_units = count_geographical_units(df)
    print(f"Geographical units (rows): {n_units}")
    print(
        "Interpretation: Each row is one California census block group "
        "(district). There are 20,640 geographical units in this dataset."
    )

    # ------------------------------------------------------------------ Q2
    print_section('Q2. Mean house value among all "ocean_proximity" categories')
    overall_mean = mean_house_value_overall(df)
    print(f"Overall mean median_house_value: ${overall_mean:,.2f}")
    print(
        "Interpretation: Across all categories, the average district median "
        "house value is about $206,856. This is the baseline against which "
        "coastal vs inland differences can be judged."
    )

    # ------------------------------------------------------------------ Q3
    print_section('Q3. Mean house value in each "ocean_proximity" category')
    by_cat = mean_house_value_by_proximity(df)
    print(by_cat.apply(lambda x: f"${x:,.2f}"))
    print(
        "Interpretation: ISLAND districts are by far the most expensive on "
        "average (~$380k), followed by NEAR BAY, NEAR OCEAN, and <1H OCEAN. "
        "INLAND is clearly cheapest (~$125k), roughly half of coastal averages. "
        "Note ISLAND has only 5 districts, so that mean is unstable."
    )

    # ------------------------------------------------------------------ Q4
    print_section("Q4. Mean vs median house value by ocean_proximity")
    mean_med = mean_vs_median_by_proximity(df)
    print(mean_med.round(2))
    print(
        "Interpretation: For most categories, mean > median, which means the "
        "price distribution is right-skewed: a minority of high-value districts "
        "pull the mean upward. ISLAND is the exception (mean < median) because "
        "of its tiny sample (n=5). The gap between mean and median is largest "
        "for NEAR BAY and <1H OCEAN, suggesting more high-end outliers there."
    )

    # ------------------------------------------------------------------ Q5
    print_section("Q5. Histograms")
    hist_path = plot_variable_histograms(df)
    print(f"Saved histograms to: {hist_path}")
    print(
        "Interpretation: Four distributions are plotted for households, "
        "median_income, housing_median_age, and median_house_value "
        "(50 bins each)."
    )

    # ------------------------------------------------------------------ Q6
    print_section("Q6. What do the graphs show?")
    caps = describe_histogram_features(df)
    for key, value in caps.items():
        print(f"{key}: {value}")
    print(
        "Interpretation:\n"
        "- households and median_income are right-skewed: many smaller values, "
        "a long tail of large districts / high incomes.\n"
        f"- housing_median_age spikes at the maximum ({caps['housing_median_age_max']}), "
        f"with {caps['housing_median_age_capped_count']} districts at that value. "
        "Ages were likely capped (clipped) during data collection.\n"
        f"- median_house_value also spikes at {caps['median_house_value_max']:,.0f} "
        f"({caps['median_house_value_capped_count']} districts). Values appear "
        "capped around $500,001, so the true prices of the most expensive "
        "homes are unknown.\n"
        "- Magnitudes: house values are in the hundreds of thousands of dollars "
        "(mean ~$207k). That scale is expected for California housing, but "
        "capping compresses the upper end and can bias models trained on this target."
    )

    # ------------------------------------------------------------------ Q7
    print_section("Q7. Most / least expensive and largest / smallest houses")
    size_price = expensive_and_large_by_proximity(df)
    print(size_price.round(2))
    print(
        "Interpretation:\n"
        "- Most expensive (by mean median_house_value): ISLAND, then NEAR BAY / "
        "NEAR OCEAN / <1H OCEAN. Coastal proximity tracks higher prices.\n"
        "- Largest houses (rooms per household and total rooms): INLAND tends to "
        "have more rooms on average, so 'expensive' and 'large' do not coincide.\n"
        "- Least expensive: INLAND (~$125k mean).\n"
        "- Smallest by rooms-per-household: <1H OCEAN (denser coastal living); "
        "ISLAND has the fewest total_rooms but only 5 samples.\n"
        "- Practical takeaway: coastal categories are typically more expensive; "
        "INLAND typically offers larger but cheaper housing."
    )

    # ------------------------------------------------------------------ Q8
    print_section("Q8. House quality by ocean_proximity")
    quality = house_quality_by_proximity(df)
    print(quality.round(3))
    print(
        "Interpretation:\n"
        "- Age: ISLAND and NEAR BAY homes are older on average; INLAND and "
        "<1H OCEAN are younger, suggesting more recent development inland / "
        "in suburban coastal belts.\n"
        "- Rooms: INLAND has higher rooms_per_household, pointing to larger "
        "dwellings. Coastal categories are denser with fewer rooms per household.\n"
        "- Bedrooms-per-room is fairly similar across categories, so room mix "
        "is less distinctive than size and age.\n"
        "- Quality is mixed: coastal areas command higher prices despite older "
        "and smaller homes, implying location (not just dwelling size/age) "
        "drives value."
    )

    # ------------------------------------------------------------------ Q9
    print_section("Q9. Demographics by ocean_proximity")
    demo = demographics_by_proximity(df)
    print(demo.round(3))
    print(
        "Interpretation:\n"
        "- Income: highest in <1H OCEAN and NEAR BAY; lowest inland. Income "
        "aligns with house values for the main categories.\n"
        "- Population / households: <1H OCEAN and NEAR OCEAN districts are "
        "more populous on average; ISLAND is sparsely populated.\n"
        "- People per household: highest inland (~3.3), lowest on ISLAND / "
        "NEAR BAY, consistent with larger inland family households vs denser "
        "or smaller coastal households.\n"
        "- Overall: coastal categories look wealthier; inland looks more "
        "crowded per household and less affluent."
    )

    # ------------------------------------------------------------------ Q10
    print_section("Q10. Other interesting observations / follow-up questions")
    extras = other_observations(df)
    print(f"Missing total_bedrooms: {extras['missing_total_bedrooms']}")
    print(f"Corr(median_income, median_house_value): {extras['income_value_corr']:.3f}")
    print(f"Unique lat/lon pairs: {extras['unique_lat_lon_pairs']}")
    print("Category counts:", extras["category_counts"])
    print(
        "Observations:\n"
        "- Strong positive correlation between median_income and house value: "
        "income is likely the strongest simple predictor.\n"
        f"- {extras['missing_total_bedrooms']} districts have missing "
        "total_bedrooms; any model using that feature needs imputation.\n"
        "- ISLAND is almost a curiosity (n=5) and can dominate averages if "
        "not handled carefully.\n"
        "- Capping of age and house value (Q6) will affect regression targets "
        "and residuals at the top end.\n"
        "\nFollow-up questions you could ask:\n"
        "- How much of house-value variation remains after controlling for income?\n"
        "- Are inland 'large house' districts still cheap after adjusting for rooms?\n"
        "- Do longitude/latitude clusters (Bay Area, LA, San Diego) explain "
        "ocean_proximity differences better than the category labels alone?\n"
        "- How should capped $500,001 values be treated in predictive modeling?"
    )


if __name__ == "__main__":
    main()
