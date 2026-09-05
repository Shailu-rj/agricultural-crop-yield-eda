import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# WEEK 2 - DETAILED EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# Agricultural Crop Yield Dataset
# Purpose:
# Detailed EDA with numerical evidence, statistical analysis,
# relationships, trends, anomalies and data-driven findings.
# ============================================================

print("=" * 80)
print("WEEK 2 - DETAILED EXPLORATORY DATA ANALYSIS")
print("=" * 80)

# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

file_path = "cleaned_crop_yield_final.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"'{file_path}' was not found. Keep this Python file in the "
        "same folder as the cleaned dataset."
    )

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Create output folder
os.makedirs("EDA_Outputs", exist_ok=True)

# ------------------------------------------------------------
# 2. BASIC DATASET OVERVIEW
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

overview = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str),
    "Missing_Values": df.isnull().sum().values,
    "Unique_Values": [df[col].nunique() for col in df.columns]
})

overview.to_csv(
    "EDA_Outputs/dataset_overview.csv",
    index=False
)

# Dataset profile
dataset_profile = pd.DataFrame({
    "Metric": [
        "Rows",
        "Columns",
        "Crops",
        "States",
        "Seasons",
        "Years",
        "Missing cells",
        "Duplicate rows"
    ],
    "Value": [
        len(df),
        len(df.columns),
        df["Crop"].nunique(),
        df["State"].nunique(),
        df["Season"].nunique(),
        df["Crop_Year"].nunique(),
        int(df.isnull().sum().sum()),
        int(df.duplicated().sum())
    ]
})

dataset_profile.to_csv(
    "EDA_Outputs/dataset_profile.csv",
    index=False
)

# ------------------------------------------------------------
# 3. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("2. STATISTICAL SUMMARY")
print("=" * 80)

numeric_cols = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

stats = df[numeric_cols].describe().T

stats["range"] = stats["max"] - stats["min"]

stats["IQR"] = (
    stats["75%"] - stats["25%"]
)

stats["skewness"] = (
    df[numeric_cols].skew()
)

stats["coefficient_of_variation_%"] = (
    df[numeric_cols].std()
    / df[numeric_cols].mean()
) * 100

stats.to_csv(
    "EDA_Outputs/statistical_summary_eda.csv"
)

print(stats.round(4))

# ------------------------------------------------------------
# 4. PERCENTILE ANALYSIS
# ------------------------------------------------------------

percentiles = df[numeric_cols].quantile(
    [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
).T

percentiles.columns = [
    "P01",
    "P05",
    "P25",
    "P50_Median",
    "P75",
    "P95",
    "P99"
]

percentiles.to_csv(
    "EDA_Outputs/percentile_analysis.csv"
)

print("\nPercentile Analysis:")
print(percentiles.round(4))

# ------------------------------------------------------------
# 5. MEAN VS MEDIAN + SKEWNESS
# ------------------------------------------------------------

distribution_analysis = []

for col in numeric_cols:

    mean_val = df[col].mean()
    median_val = df[col].median()
    skew_val = df[col].skew()

    if abs(skew_val) < 0.5:
        interpretation = "Approximately symmetric"

    elif skew_val >= 0.5:
        interpretation = "Positively / right skewed"

    else:
        interpretation = "Negatively / left skewed"

    distribution_analysis.append({
        "Variable": col,
        "Mean": mean_val,
        "Median": median_val,
        "Mean_Minus_Median": mean_val - median_val,
        "Skewness": skew_val,
        "Interpretation": interpretation
    })

distribution_analysis = pd.DataFrame(
    distribution_analysis
)

distribution_analysis.to_csv(
    "EDA_Outputs/mean_median_skewness_analysis.csv",
    index=False
)

# ------------------------------------------------------------
# 6. DISTRIBUTION HISTOGRAMS
# ------------------------------------------------------------

print("\nCreating distribution plots...")

for col in numeric_cols:

    plt.figure(figsize=(9, 6))

    sns.histplot(
        df[col],
        bins=40,
        kde=True
    )

    plt.title(
        f"Distribution of {col}"
    )

    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"EDA_Outputs/distribution_{col}.png",
        dpi=300
    )

    plt.close()

print("Distribution plots completed.")

# ------------------------------------------------------------
# 7. BOXPLOTS + IQR OUTLIERS
# ------------------------------------------------------------

print("\nCreating boxplots and calculating IQR outliers...")

outlier_rows = []

for col in numeric_cols:

    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_mask = (
        (df[col] < lower_bound)
        |
        (df[col] > upper_bound)
    )

    count = int(outlier_mask.sum())

    outlier_rows.append({
        "Variable": col,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr,
        "Lower_Bound": lower_bound,
        "Upper_Bound": upper_bound,
        "Outlier_Count": count,
        "Outlier_Percentage": (
            count / len(df)
        ) * 100
    })

    plt.figure(figsize=(9, 5))

    sns.boxplot(
        x=df[col]
    )

    plt.title(
        f"Boxplot of {col}"
    )

    plt.xlabel(col)

    plt.tight_layout()

    plt.savefig(
        f"EDA_Outputs/boxplot_{col}.png",
        dpi=300
    )

    plt.close()

outlier_analysis = pd.DataFrame(
    outlier_rows
)

outlier_analysis.to_csv(
    "EDA_Outputs/iqr_outlier_analysis.csv",
    index=False
)

print(outlier_analysis.round(4))

# ------------------------------------------------------------
# 8. CATEGORICAL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("3. CATEGORICAL ANALYSIS")
print("=" * 80)

# ------------------------------------------------------------
# Crop frequency
# ------------------------------------------------------------

crop_frequency = (
    df["Crop"]
    .value_counts()
    .head(10)
)

crop_frequency.to_csv(
    "EDA_Outputs/top_10_crop_frequency.csv"
)

plt.figure(figsize=(11, 6))

crop_frequency.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 Most Frequently Recorded Crops"
)

plt.xlabel("Number of Records")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/top_10_crop_frequency.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# State frequency
# ------------------------------------------------------------

state_frequency = (
    df["State"]
    .value_counts()
    .head(10)
)

state_frequency.to_csv(
    "EDA_Outputs/top_10_state_frequency.csv"
)

plt.figure(figsize=(11, 6))

state_frequency.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 States by Number of Records"
)

plt.xlabel("Number of Records")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/top_10_state_frequency.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# Season frequency
# ------------------------------------------------------------

season_frequency = (
    df["Season"]
    .value_counts()
)

season_frequency.to_csv(
    "EDA_Outputs/season_frequency.csv"
)

plt.figure(figsize=(9, 6))

season_frequency.plot(
    kind="bar"
)

plt.title(
    "Distribution of Agricultural Seasons"
)

plt.xlabel("Season")
plt.ylabel("Number of Records")

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/season_distribution.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 9. CROP PRODUCTION + PRODUCTION SHARE
# ------------------------------------------------------------

total_production = df["Production"].sum()

crop_production_all = (
    df.groupby("Crop")["Production"]
    .sum()
    .sort_values(ascending=False)
)

crop_production = (
    crop_production_all
    .head(10)
)

crop_production_share = (
    crop_production
    / total_production
    * 100
)

top_crop_table = pd.DataFrame({
    "Total_Production":
        crop_production,

    "Production_Share_%":
        crop_production_share
})

top_crop_table.to_csv(
    "EDA_Outputs/top_10_crops_production_with_share.csv"
)

plt.figure(figsize=(11, 6))

crop_production.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 Crops by Total Production"
)

plt.xlabel("Total Production")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/top_10_crops_production_eda.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 10. STATE PRODUCTION + PRODUCTION SHARE
# ------------------------------------------------------------

state_production_all = (
    df.groupby("State")["Production"]
    .sum()
    .sort_values(ascending=False)
)

state_production = (
    state_production_all
    .head(10)
)

state_production_share = (
    state_production
    / total_production
    * 100
)

top_state_table = pd.DataFrame({
    "Total_Production":
        state_production,

    "Production_Share_%":
        state_production_share
})

top_state_table.to_csv(
    "EDA_Outputs/top_10_states_production_with_share.csv"
)

plt.figure(figsize=(11, 6))

state_production.sort_values().plot(
    kind="barh"
)

plt.title(
    "Top 10 States by Total Crop Production"
)

plt.xlabel("Total Production")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/top_10_states_production_eda.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 11. SEASON-WISE YIELD ANALYSIS
# ------------------------------------------------------------

season_yield_stats = (
    df.groupby("Season")["Yield"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
    .sort_values(
        "mean",
        ascending=False
    )
)

season_yield_stats.to_csv(
    "EDA_Outputs/season_yield_detailed.csv"
)

season_yield = (
    season_yield_stats["mean"]
)

plt.figure(figsize=(9, 6))

season_yield.plot(
    kind="bar"
)

plt.title(
    "Average Crop Yield by Season"
)

plt.xlabel("Season")
plt.ylabel("Average Yield")

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/average_yield_by_season_eda.png",
    dpi=300
)

plt.close()

best_season = (
    season_yield.idxmax()
)

worst_season = (
    season_yield.idxmin()
)

best_season_yield = (
    season_yield[best_season]
)

worst_season_yield = (
    season_yield[worst_season]
)

season_difference = (
    best_season_yield
    - worst_season_yield
)

if worst_season_yield != 0:

    season_difference_pct = (
        season_difference
        / worst_season_yield
    ) * 100

else:

    season_difference_pct = np.nan

season_comparison = pd.DataFrame({
    "Metric": [
        "Best season",
        "Best season average yield",
        "Worst season",
        "Worst season average yield",
        "Absolute difference",
        "Percentage difference relative to worst"
    ],

    "Value": [
        best_season,
        best_season_yield,
        worst_season,
        worst_season_yield,
        season_difference,
        season_difference_pct
    ]
})

season_comparison.to_csv(
    "EDA_Outputs/best_vs_worst_season.csv",
    index=False
)

print("\nAverage Yield by Season:")
print(season_yield_stats.round(4))

# ------------------------------------------------------------
# 12. CROP-WISE YIELD ANALYSIS
# ------------------------------------------------------------

crop_yield_stats = (
    df.groupby("Crop")["Yield"]
    .agg([
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max"
    ])
)

crop_yield_stats.to_csv(
    "EDA_Outputs/all_crop_yield_statistics.csv"
)

# Minimum 20 observations
eligible_crop_yield = (
    crop_yield_stats[
        crop_yield_stats["count"] >= 20
    ]
    .sort_values(
        "mean",
        ascending=False
    )
)

top_10_crop_yield = (
    eligible_crop_yield
    .head(10)
)

top_10_crop_yield.to_csv(
    "EDA_Outputs/top_10_crops_average_yield.csv"
)

plt.figure(figsize=(11, 6))

top_10_crop_yield["mean"].sort_values().plot(
    kind="barh"
)

plt.title(
    "Top Crops by Average Yield (Minimum 20 Records)"
)

plt.xlabel("Average Yield")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/top_10_crops_average_yield.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 13. RELATIONSHIP ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("4. RELATIONSHIP ANALYSIS")
print("=" * 80)

relationships = [
    ("Area", "Production"),
    ("Fertilizer", "Production"),
    ("Pesticide", "Production"),
    ("Annual_Rainfall", "Yield"),
    ("Production", "Yield")
]

relationship_results = []

for x, y in relationships:

    pair = df[
        [x, y]
    ].dropna()

    r = pair[x].corr(
        pair[y]
    )

    # Simple linear regression
    slope, intercept = np.polyfit(
        pair[x],
        pair[y],
        1
    )

    predicted = (
        slope * pair[x]
        + intercept
    )

    ss_res = (
        (pair[y] - predicted) ** 2
    ).sum()

    ss_tot = (
        (pair[y] - pair[y].mean()) ** 2
    ).sum()

    if ss_tot != 0:

        r_squared = (
            1
            - ss_res / ss_tot
        )

    else:

        r_squared = np.nan

    relationship_results.append({
        "Variable_X": x,
        "Variable_Y": y,
        "Pearson_Correlation": r,
        "R_Squared": r_squared,
        "Slope": slope,
        "Intercept": intercept,
        "Absolute_Correlation": abs(r)
    })

    plt.figure(figsize=(9, 6))

    sns.regplot(
        data=pair,
        x=x,
        y=y,
        scatter_kws={
            "alpha": 0.35
        }
    )

    plt.title(
        f"{x} vs {y}\n"
        f"Pearson r = {r:.3f}, "
        f"R² = {r_squared:.3f}"
    )

    plt.xlabel(x)
    plt.ylabel(y)

    plt.tight_layout()

    filename = (
        f"EDA_Outputs/"
        f"{x}_vs_{y}_regression.png"
    )

    plt.savefig(
        filename,
        dpi=300
    )

    plt.close()

relationship_results = pd.DataFrame(
    relationship_results
)

relationship_results = (
    relationship_results
    .sort_values(
        "Absolute_Correlation",
        ascending=False
    )
)

relationship_results.to_csv(
    "EDA_Outputs/relationship_analysis_regression.csv",
    index=False
)

print("\nRelationship Analysis:")
print(
    relationship_results.round(4)
)

# ------------------------------------------------------------
# 14. CORRELATION MATRIX
# ------------------------------------------------------------

print("\nCalculating correlations...")

correlation = (
    df[numeric_cols]
    .corr()
)

correlation.to_csv(
    "EDA_Outputs/correlation_matrix_eda.csv"
)

print("\nCorrelation Matrix:")
print(
    correlation.round(3)
)

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title(
    "Correlation Heatmap of Numerical Variables"
)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/correlation_heatmap_eda.png",
    dpi=300
)

plt.close()

# Strongest correlations

corr_pairs = correlation.where(
    np.triu(
        np.ones(
            correlation.shape
        ),
        k=1
    ).astype(bool)
)

corr_pairs = (
    corr_pairs
    .stack()
    .reset_index()
)

corr_pairs.columns = [
    "Variable_1",
    "Variable_2",
    "Correlation"
]

corr_pairs["Absolute_Correlation"] = (
    corr_pairs["Correlation"]
    .abs()
)

corr_pairs = (
    corr_pairs
    .sort_values(
        "Absolute_Correlation",
        ascending=False
    )
)

corr_pairs.to_csv(
    "EDA_Outputs/strongest_correlations.csv",
    index=False
)

print("\nStrongest Correlations:")
print(
    corr_pairs.head(10).round(4)
)

# ------------------------------------------------------------
# 15. YEAR-WISE TREND ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("5. YEAR-WISE TREND ANALYSIS")
print("=" * 80)

year_analysis = (
    df.groupby("Crop_Year")
    .agg(
        Total_Production=(
            "Production",
            "sum"
        ),

        Average_Yield=(
            "Yield",
            "mean"
        ),

        Median_Yield=(
            "Yield",
            "median"
        ),

        Average_Rainfall=(
            "Annual_Rainfall",
            "mean"
        ),

        Total_Area=(
            "Area",
            "sum"
        ),

        Record_Count=(
            "Crop",
            "size"
        )
    )
    .reset_index()
    .sort_values(
        "Crop_Year"
    )
)

# Year-over-year changes

year_analysis[
    "Production_YoY_%"
] = (
    year_analysis[
        "Total_Production"
    ].pct_change()
    * 100
)

year_analysis[
    "Yield_YoY_%"
] = (
    year_analysis[
        "Average_Yield"
    ].pct_change()
    * 100
)

year_analysis[
    "Rainfall_YoY_%"
] = (
    year_analysis[
        "Average_Rainfall"
    ].pct_change()
    * 100
)

year_analysis.to_csv(
    "EDA_Outputs/year_wise_detailed_analysis.csv",
    index=False
)

print("\nYear-wise Detailed Analysis:")
print(
    year_analysis.round(4)
)

# Production trend

plt.figure(figsize=(11, 6))

plt.plot(
    year_analysis["Crop_Year"],
    year_analysis["Total_Production"],
    marker="o"
)

plt.title(
    "Year-wise Total Crop Production"
)

plt.xlabel("Year")
plt.ylabel("Total Production")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/year_wise_production_trend.png",
    dpi=300
)

plt.close()

# Yield trend

plt.figure(figsize=(11, 6))

plt.plot(
    year_analysis["Crop_Year"],
    year_analysis["Average_Yield"],
    marker="o"
)

plt.title(
    "Year-wise Average Crop Yield"
)

plt.xlabel("Year")
plt.ylabel("Average Yield")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/year_wise_yield_trend.png",
    dpi=300
)

plt.close()

# Rainfall trend

plt.figure(figsize=(11, 6))

plt.plot(
    year_analysis["Crop_Year"],
    year_analysis["Average_Rainfall"],
    marker="o"
)

plt.title(
    "Year-wise Average Annual Rainfall"
)

plt.xlabel("Year")
plt.ylabel("Average Rainfall")

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/year_wise_rainfall_trend.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 16. FIRST VS LAST YEAR ANALYSIS
# ------------------------------------------------------------

first_year = (
    year_analysis.iloc[0]
)

last_year = (
    year_analysis.iloc[-1]
)

def percentage_change(
    first,
    last
):

    if first == 0:
        return np.nan

    return (
        (last - first)
        / first
    ) * 100

trend_summary = pd.DataFrame({
    "Metric": [
        "Production",
        "Average Yield",
        "Average Rainfall",
        "Total Area"
    ],

    "First_Year": [
        first_year[
            "Total_Production"
        ],

        first_year[
            "Average_Yield"
        ],

        first_year[
            "Average_Rainfall"
        ],

        first_year[
            "Total_Area"
        ]
    ],

    "Last_Year": [
        last_year[
            "Total_Production"
        ],

        last_year[
            "Average_Yield"
        ],

        last_year[
            "Average_Rainfall"
        ],

        last_year[
            "Total_Area"
        ]
    ],

    "Percentage_Change": [
        percentage_change(
            first_year[
                "Total_Production"
            ],
            last_year[
                "Total_Production"
            ]
        ),

        percentage_change(
            first_year[
                "Average_Yield"
            ],
            last_year[
                "Average_Yield"
            ]
        ),

        percentage_change(
            first_year[
                "Average_Rainfall"
            ],
            last_year[
                "Average_Rainfall"
            ]
        ),

        percentage_change(
            first_year[
                "Total_Area"
            ],
            last_year[
                "Total_Area"
            ]
        )
    ]
})

trend_summary.to_csv(
    "EDA_Outputs/first_vs_last_year_trends.csv",
    index=False
)

# ------------------------------------------------------------
# 17. ZERO PRODUCTION ANALYSIS
# ------------------------------------------------------------

zero_production = (
    df[
        df["Production"] == 0
    ].copy()
)

zero_production_count = (
    len(zero_production)
)

zero_production_percentage = (
    zero_production_count
    / len(df)
    * 100
)

zero_and_zero_yield = len(
    df[
        (df["Production"] == 0)
        &
        (df["Yield"] == 0)
    ]
)

zero_production_positive_yield = len(
    df[
        (df["Production"] == 0)
        &
        (df["Yield"] > 0)
    ]
)

positive_production_zero_yield = len(
    df[
        (df["Production"] > 0)
        &
        (df["Yield"] == 0)
    ]
)

zero_production_summary = pd.DataFrame({
    "Metric": [
        "Zero Production Records",
        "Percentage of Dataset",
        "Zero Production and Zero Yield",
        "Zero Production but Positive Yield",
        "Positive Production but Zero Yield"
    ],

    "Value": [
        zero_production_count,
        zero_production_percentage,
        zero_and_zero_yield,
        zero_production_positive_yield,
        positive_production_zero_yield
    ]
})

zero_production_summary.to_csv(
    "EDA_Outputs/zero_production_analysis.csv",
    index=False
)

print(
    "\nZero Production Records:",
    zero_production_count
)

print(
    "Percentage:",
    round(
        zero_production_percentage,
        2
    ),
    "%"
)

# Zero production by crop

zero_by_crop = (
    zero_production[
        "Crop"
    ]
    .value_counts()
    .head(10)
)

zero_by_crop.to_csv(
    "EDA_Outputs/zero_production_by_crop.csv"
)

# Zero production by state

zero_by_state = (
    zero_production[
        "State"
    ]
    .value_counts()
    .head(10)
)

zero_by_state.to_csv(
    "EDA_Outputs/zero_production_by_state.csv"
)

# Zero production by season

zero_by_season = (
    zero_production[
        "Season"
    ]
    .value_counts()
)

zero_by_season.to_csv(
    "EDA_Outputs/zero_production_by_season.csv"
)

# ------------------------------------------------------------
# 18. DATA CONSISTENCY CHECKS
# ------------------------------------------------------------

positive_production_zero_yield_df = df[
    (df["Production"] > 0)
    &
    (df["Yield"] == 0)
].copy()

zero_production_positive_yield_df = df[
    (df["Production"] == 0)
    &
    (df["Yield"] > 0)
].copy()

positive_production_zero_yield_df.to_csv(
    "EDA_Outputs/positive_production_zero_yield.csv",
    index=False
)

zero_production_positive_yield_df.to_csv(
    "EDA_Outputs/zero_production_positive_yield.csv",
    index=False
)

# ------------------------------------------------------------
# 19. YIELD ANOMALY ANALYSIS
# ------------------------------------------------------------

yield_q1 = (
    df["Yield"].quantile(0.25)
)

yield_q3 = (
    df["Yield"].quantile(0.75)
)

yield_iqr = (
    yield_q3 - yield_q1
)

yield_lower = (
    yield_q1
    - 1.5 * yield_iqr
)

yield_upper = (
    yield_q3
    + 1.5 * yield_iqr
)

high_yield = df[
    df["Yield"] > yield_upper
].copy()

low_yield = df[
    df["Yield"] < yield_lower
].copy()

high_yield.to_csv(
    "EDA_Outputs/high_yield_anomalies.csv",
    index=False
)

low_yield.to_csv(
    "EDA_Outputs/low_yield_anomalies.csv",
    index=False
)

yield_anomaly_summary = pd.DataFrame({
    "Metric": [
        "Q1",
        "Q3",
        "IQR",
        "Lower IQR Bound",
        "Upper IQR Bound",
        "High Yield Anomalies",
        "Low Yield Anomalies",
        "High Yield Percentage",
        "Low Yield Percentage"
    ],

    "Value": [
        yield_q1,
        yield_q3,
        yield_iqr,
        yield_lower,
        yield_upper,
        len(high_yield),
        len(low_yield),
        len(high_yield)
        / len(df)
        * 100,
        len(low_yield)
        / len(df)
        * 100
    ]
})

yield_anomaly_summary.to_csv(
    "EDA_Outputs/yield_anomaly_summary.csv",
    index=False
)

print(
    "\nYield anomaly analysis:"
)

print(
    yield_anomaly_summary.round(4)
)

# ------------------------------------------------------------
# 20. EXTREME OBSERVATIONS
# ------------------------------------------------------------

extreme_records = {}

for col in numeric_cols:

    highest_idx = (
        df[col].idxmax()
    )

    lowest_idx = (
        df[col].idxmin()
    )

    extreme_records[col] = {

        "Highest_Value":
            df.loc[
                highest_idx,
                col
            ],

        "Highest_Crop":
            df.loc[
                highest_idx,
                "Crop"
            ],

        "Highest_State":
            df.loc[
                highest_idx,
                "State"
            ],

        "Highest_Year":
            df.loc[
                highest_idx,
                "Crop_Year"
            ],

        "Lowest_Value":
            df.loc[
                lowest_idx,
                col
            ],

        "Lowest_Crop":
            df.loc[
                lowest_idx,
                "Crop"
            ],

        "Lowest_State":
            df.loc[
                lowest_idx,
                "State"
            ],

        "Lowest_Year":
            df.loc[
                lowest_idx,
                "Crop_Year"
            ]
    }

extreme_table = (
    pd.DataFrame(
        extreme_records
    ).T
)

extreme_table.to_csv(
    "EDA_Outputs/extreme_observations.csv"
)

# ------------------------------------------------------------
# 21. WORKED NUMERICAL EXAMPLES
# ------------------------------------------------------------

top_crop = (
    crop_production.index[0]
)

top_crop_value = (
    crop_production.iloc[0]
)

top_crop_share = (
    top_crop_value
    / total_production
    * 100
)

area_production_r = (
    correlation.loc[
        "Area",
        "Production"
    ]
)

production_yield_r = (
    correlation.loc[
        "Production",
        "Yield"
    ]
)

yield_iqr_example = (
    yield_q3 - yield_q1
)

yield_upper_example = (
    yield_q3
    + 1.5 * yield_iqr_example
)

production_mean = (
    df["Production"].mean()
)

production_median = (
    df["Production"].median()
)

worked_examples = pd.DataFrame({

    "Example": [

        "Top crop production share",

        "Area-Production correlation",

        "Production-Yield correlation",

        "Yield IQR",

        "Yield upper IQR threshold",

        "Production mean",

        "Production median",

        "Production mean-minus-median"
    ],

    "Value": [

        top_crop_share,

        area_production_r,

        production_yield_r,

        yield_iqr_example,

        yield_upper_example,

        production_mean,

        production_median,

        production_mean
        - production_median
    ],

    "Explanation": [

        f"({top_crop_value:,.2f} / "
        f"{total_production:,.2f}) x 100",

        "Pearson correlation between "
        "Area and Production",

        "Pearson correlation between "
        "Production and Yield",

        f"Q3 - Q1 = "
        f"{yield_q3:.4f} - "
        f"{yield_q1:.4f}",

        f"Q3 + 1.5 x IQR = "
        f"{yield_q3:.4f} + "
        f"1.5 x {yield_iqr_example:.4f}",

        "Arithmetic mean of Production",

        "50th percentile of Production",

        "Mean - Median"
    ]
})

worked_examples.to_csv(
    "EDA_Outputs/worked_numerical_examples.csv",
    index=False
)

# ------------------------------------------------------------
# 22. PRODUCTION CONCENTRATION
# ------------------------------------------------------------

top_5_crop_share = (
    crop_production_all
    .head(5)
    .sum()
    / total_production
    * 100
)

top_10_crop_share = (
    crop_production_all
    .head(10)
    .sum()
    / total_production
    * 100
)

top_5_state_share = (
    state_production_all
    .head(5)
    .sum()
    / total_production
    * 100
)

top_10_state_share = (
    state_production_all
    .head(10)
    .sum()
    / total_production
    * 100
)

concentration = pd.DataFrame({

    "Group": [
        "Top 5 crops",
        "Top 10 crops",
        "Top 5 states",
        "Top 10 states"
    ],

    "Production_Share_%": [
        top_5_crop_share,
        top_10_crop_share,
        top_5_state_share,
        top_10_state_share
    ]
})

concentration.to_csv(
    "EDA_Outputs/production_concentration.csv",
    index=False
)

# ------------------------------------------------------------
# 23. ORIGINAL DATA-DRIVEN FINDINGS
# ------------------------------------------------------------

strongest_positive = (
    corr_pairs
    .sort_values(
        "Correlation",
        ascending=False
    )
    .iloc[0]
)

strongest_negative = (
    corr_pairs
    .sort_values(
        "Correlation",
        ascending=True
    )
    .iloc[0]
)

highest_cv_row = (
    stats[
        "coefficient_of_variation_%"
    ]
    .idxmax()
)

highest_cv_value = (
    stats.loc[
        highest_cv_row,
        "coefficient_of_variation_%"
    ]
)

findings = [

    f"The final cleaned dataset contains "
    f"{len(df):,} observations across "
    f"{len(df.columns)} variables, covering "
    f"{df['Crop'].nunique()} crops, "
    f"{df['State'].nunique()} states, "
    f"{df['Season'].nunique()} seasons and "
    f"{df['Crop_Year'].nunique()} crop years.",

    f"Data-quality checks found "
    f"{int(df.isnull().sum().sum())} missing cells "
    f"and {int(df.duplicated().sum())} duplicate rows, "
    f"indicating that the Week 1 cleaned dataset was "
    f"structurally consistent for EDA.",

    f"Production has a mean of "
    f"{production_mean:,.3f} and a median of "
    f"{production_median:,.3f}. The mean is higher "
    f"than the median, which is consistent with a "
    f"right-skewed production distribution.",

    f"{highest_cv_row} has the highest coefficient "
    f"of variation among the analysed numerical "
    f"variables at approximately "
    f"{highest_cv_value:.2f}%, indicating particularly "
    f"high relative variability.",

    f"{top_crop} is the highest-producing crop in "
    f"the dataset with total production of "
    f"{top_crop_value:,.2f}, contributing approximately "
    f"{top_crop_share:.2f}% of total recorded production.",

    f"The top five crops together account for "
    f"approximately {top_5_crop_share:.2f}% of total "
    f"production, showing that production is "
    f"concentrated among a smaller group of crops.",

    f"The highest-producing state is "
    f"{state_production.index[0]} with total production "
    f"of {state_production.iloc[0]:,.2f}, representing "
    f"{state_production_share.iloc[0]:.2f}% of total "
    f"production.",

    f"{best_season} has the highest average yield at "
    f"{best_season_yield:.3f}, while {worst_season} "
    f"has the lowest average yield at "
    f"{worst_season_yield:.3f}. The absolute gap is "
    f"{season_difference:.3f}, approximately "
    f"{season_difference_pct:.2f}% relative to the "
    f"lower-yield season.",

    f"Area and Production have a Pearson correlation "
    f"of {area_production_r:.3f}, indicating a strong "
    f"positive association. Larger cultivated areas "
    f"therefore tend to be associated with greater "
    f"total production in these observations.",

    f"Production and Yield have a weaker positive "
    f"correlation of {production_yield_r:.3f}. This "
    f"indicates that higher total production does not "
    f"automatically imply proportionally higher yield.",

    f"The strongest positive numerical relationship "
    f"is between {strongest_positive['Variable_1']} "
    f"and {strongest_positive['Variable_2']} with "
    f"r = {strongest_positive['Correlation']:.3f}. "
    f"This is an association and should not be "
    f"interpreted as causation.",

    f"The strongest negative numerical relationship "
    f"is between {strongest_negative['Variable_1']} "
    f"and {strongest_negative['Variable_2']} with "
    f"r = {strongest_negative['Correlation']:.3f}. "
    f"The negative sign indicates an inverse "
    f"statistical association.",

    f"There are {zero_production_count:,} zero-production "
    f"observations, representing "
    f"{zero_production_percentage:.2f}% of the dataset. "
    f"These were not automatically treated as errors "
    f"because zero output can represent a genuine "
    f"agricultural outcome.",

    f"There are {positive_production_zero_yield:,} "
    f"observations with positive Production but zero "
    f"Yield. These observations deserve attention "
    f"when interpreting productivity-related results.",

    f"The IQR-based yield analysis identified "
    f"{len(high_yield):,} high-yield observations and "
    f"{len(low_yield):,} low-yield observations using "
    f"statistical thresholds rather than arbitrary "
    f"cut-offs.",

    f"From the first recorded year "
    f"({int(first_year['Crop_Year'])}) to the last "
    f"recorded year ({int(last_year['Crop_Year'])}), "
    f"total production changed by "
    f"{trend_summary.loc[0, 'Percentage_Change']:.2f}%, "
    f"while average yield changed by "
    f"{trend_summary.loc[1, 'Percentage_Change']:.2f}%. "
    f"These changes should be interpreted within the "
    f"years and observations represented in the dataset.",

    f"The top ten states account for approximately "
    f"{top_10_state_share:.2f}% of total production. "
    f"This means overall production patterns can be "
    f"strongly influenced by a relatively small number "
    f"of high-producing states.",

    "Correlation and regression results describe "
    "statistical relationships within the available "
    "observations. They should not be interpreted as "
    "proof that one agricultural factor directly "
    "causes another."
]

with open(
    "EDA_Outputs/detailed_eda_findings.txt",
    "w",
    encoding="utf-8"
) as file:

    for i, finding in enumerate(
        findings,
        1
    ):

        file.write(
            f"{i}. {finding}\n"
        )

# ------------------------------------------------------------
# 24. REPORT-READY SUMMARY TABLE
# ------------------------------------------------------------

summary_table = pd.DataFrame({

    "Analysis": [

        "Dataset size",
        "Missing values",
        "Duplicate rows",
        "Number of crops",
        "Number of states",
        "Number of seasons",
        "Zero-production records",
        "Zero-production percentage",
        "High-yield IQR anomalies",
        "Low-yield IQR anomalies",
        "Best-yield season",
        "Best-yield season average",
        "Worst-yield season",
        "Worst-yield season average",
        "Strongest positive correlation",
        "Strongest negative correlation",
        "Top crop by production",
        "Top state by production"
    ],

    "Result": [

        f"{len(df):,} rows x "
        f"{len(df.columns)} columns",

        int(
            df.isnull()
            .sum()
            .sum()
        ),

        int(
            df.duplicated()
            .sum()
        ),

        df["Crop"].nunique(),

        df["State"].nunique(),

        df["Season"].nunique(),

        zero_production_count,

        round(
            zero_production_percentage,
            2
        ),

        len(high_yield),

        len(low_yield),

        best_season,

        round(
            best_season_yield,
            4
        ),

        worst_season,

        round(
            worst_season_yield,
            4
        ),

        f"{strongest_positive['Variable_1']} vs "
        f"{strongest_positive['Variable_2']} = "
        f"{strongest_positive['Correlation']:.3f}",

        f"{strongest_negative['Variable_1']} vs "
        f"{strongest_negative['Variable_2']} = "
        f"{strongest_negative['Correlation']:.3f}",

        top_crop,

        state_production.index[0]
    ]
})

summary_table.to_csv(
    "EDA_Outputs/eda_report_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 25. FINAL CONSOLE SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("DETAILED EDA COMPLETED SUCCESSFULLY!")
print("=" * 80)

print("\nDataset:")
print(
    f"- {len(df):,} rows"
)
print(
    f"- {len(df.columns)} columns"
)
print(
    f"- {df['Crop'].nunique()} crops"
)
print(
    f"- {df['State'].nunique()} states"
)
print(
    f"- {df['Season'].nunique()} seasons"
)

print("\nData Quality:")
print(
    f"- Missing values: "
    f"{int(df.isnull().sum().sum())}"
)
print(
    f"- Duplicate rows: "
    f"{int(df.duplicated().sum())}"
)

print("\nImportant Findings:")
print(
    f"- Zero production: "
    f"{zero_production_count} "
    f"({zero_production_percentage:.2f}%)"
)

print(
    f"- Best yield season: "
    f"{best_season} "
    f"({best_season_yield:.3f})"
)

print(
    f"- Top crop: "
    f"{top_crop}"
)

print(
    f"- Top state: "
    f"{state_production.index[0]}"
)

print(
    f"- Strongest positive correlation: "
    f"{strongest_positive['Variable_1']} vs "
    f"{strongest_positive['Variable_2']} "
    f"({strongest_positive['Correlation']:.3f})"
)

print(
    f"- Strongest negative correlation: "
    f"{strongest_negative['Variable_1']} vs "
    f"{strongest_negative['Variable_2']} "
    f"({strongest_negative['Correlation']:.3f})"
)

print("\nMajor outputs generated:")
print("- Statistical summaries")
print("- Percentile analysis")
print("- Mean vs median analysis")
print("- Skewness analysis")
print("- Coefficient of variation")
print("- Distribution plots")
print("- Boxplots")
print("- IQR outlier analysis")
print("- Crop frequency analysis")
print("- State frequency analysis")
print("- Season analysis")
print("- Production share analysis")
print("- Season-wise yield analysis")
print("- Crop-wise yield analysis")
print("- Regression relationship plots")
print("- R-squared analysis")
print("- Correlation matrix")
print("- Correlation ranking")
print("- Year-wise trends")
print("- Year-over-year changes")
print("- Zero-production analysis")
print("- Data consistency checks")
print("- Yield anomaly analysis")
print("- Extreme observations")
print("- Worked numerical examples")
print("- Production concentration")
print("- Detailed data-driven findings")
print("- Report-ready summary")

print("\nAll outputs saved inside:")
print("EDA_Outputs")

print("\nWeek 2 Detailed EDA is ready for report preparation.")
print("=" * 80)