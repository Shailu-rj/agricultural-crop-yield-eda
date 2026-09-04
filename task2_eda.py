import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# WEEK 2 - EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================

print("=" * 70)
print("WEEK 2 - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD CLEANED DATASET
# ------------------------------------------------------------

file_path = "cleaned_crop_yield_final.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# Create folder for EDA outputs
os.makedirs("EDA_Outputs", exist_ok=True)

# ------------------------------------------------------------
# 2. BASIC DATASET OVERVIEW
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. DATASET OVERVIEW")
print("=" * 70)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Save basic information
overview = pd.DataFrame({
    "Column": df.columns,
    "Data_Type": df.dtypes.astype(str),
    "Missing_Values": df.isnull().sum().values,
    "Unique_Values": [df[col].nunique() for col in df.columns]
})

overview.to_csv("EDA_Outputs/dataset_overview.csv", index=False)

# ------------------------------------------------------------
# 3. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. STATISTICAL SUMMARY")
print("=" * 70)

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

stats.to_csv("EDA_Outputs/statistical_summary_eda.csv")

print(stats)

# ------------------------------------------------------------
# 4. DISTRIBUTION HISTOGRAMS
# ------------------------------------------------------------

print("\nCreating distribution plots...")

for col in numeric_cols:

    plt.figure(figsize=(9, 6))

    sns.histplot(df[col], bins=40, kde=True)

    plt.title(f"Distribution of {col}")
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
# 5. BOXPLOTS
# ------------------------------------------------------------

print("\nCreating boxplots...")

for col in numeric_cols:

    plt.figure(figsize=(9, 5))

    sns.boxplot(x=df[col])

    plt.title(f"Boxplot of {col}")
    plt.xlabel(col)

    plt.tight_layout()

    plt.savefig(
        f"EDA_Outputs/boxplot_{col}.png",
        dpi=300
    )

    plt.close()

print("Boxplots completed.")

# ------------------------------------------------------------
# 6. CATEGORICAL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. CATEGORICAL ANALYSIS")
print("=" * 70)

# Crop frequency
crop_frequency = df["Crop"].value_counts().head(10)

crop_frequency.to_csv(
    "EDA_Outputs/top_10_crop_frequency.csv"
)

plt.figure(figsize=(11, 6))

crop_frequency.sort_values().plot(kind="barh")

plt.title("Top 10 Most Frequently Recorded Crops")
plt.xlabel("Number of Records")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/top_10_crop_frequency.png",
    dpi=300
)
plt.close()

# State frequency
state_frequency = df["State"].value_counts().head(10)

state_frequency.to_csv(
    "EDA_Outputs/top_10_state_frequency.csv"
)

plt.figure(figsize=(11, 6))

state_frequency.sort_values().plot(kind="barh")

plt.title("Top 10 States by Number of Records")
plt.xlabel("Number of Records")
plt.ylabel("State")

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/top_10_state_frequency.png",
    dpi=300
)
plt.close()

# Season frequency
season_frequency = df["Season"].value_counts()

season_frequency.to_csv(
    "EDA_Outputs/season_frequency.csv"
)

plt.figure(figsize=(9, 6))

season_frequency.plot(kind="bar")

plt.title("Distribution of Agricultural Seasons")
plt.xlabel("Season")
plt.ylabel("Number of Records")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/season_distribution.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 7. AVERAGE YIELD BY SEASON
# ------------------------------------------------------------

season_yield = (
    df.groupby("Season")["Yield"]
    .mean()
    .sort_values(ascending=False)
)

season_yield.to_csv(
    "EDA_Outputs/average_yield_by_season.csv"
)

plt.figure(figsize=(9, 6))

season_yield.plot(kind="bar")

plt.title("Average Crop Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield")
plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/average_yield_by_season_eda.png",
    dpi=300
)
plt.close()

print("\nAverage Yield by Season:")
print(season_yield)

# ------------------------------------------------------------
# 8. AVERAGE YIELD BY TOP CROPS
# ------------------------------------------------------------

crop_yield = (
    df.groupby("Crop")["Yield"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

crop_yield.to_csv(
    "EDA_Outputs/top_10_crops_average_yield.csv"
)

plt.figure(figsize=(11, 6))

crop_yield.sort_values().plot(kind="barh")

plt.title("Top 10 Crops by Average Yield")
plt.xlabel("Average Yield")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/top_10_crops_average_yield.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 9. PRODUCTION BY CROP
# ------------------------------------------------------------

crop_production = (
    df.groupby("Crop")["Production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

crop_production.to_csv(
    "EDA_Outputs/top_10_crops_production_eda.csv"
)

plt.figure(figsize=(11, 6))

crop_production.sort_values().plot(kind="barh")

plt.title("Top 10 Crops by Total Production")
plt.xlabel("Total Production")
plt.ylabel("Crop")

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/top_10_crops_production_eda.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 10. PRODUCTION BY STATE
# ------------------------------------------------------------

state_production = (
    df.groupby("State")["Production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

state_production.to_csv(
    "EDA_Outputs/top_10_states_production_eda.csv"
)

plt.figure(figsize=(11, 6))

state_production.sort_values().plot(kind="barh")

plt.title("Top 10 States by Total Crop Production")
plt.xlabel("Total Production")
plt.ylabel("State")

plt.tight_layout()
plt.savefig(
    "EDA_Outputs/top_10_states_production_eda.png",
    dpi=300
)
plt.close()

# ------------------------------------------------------------
# 11. RELATIONSHIP ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. RELATIONSHIP ANALYSIS")
print("=" * 70)

relationships = [
    ("Area", "Production"),
    ("Fertilizer", "Production"),
    ("Pesticide", "Production"),
    ("Annual_Rainfall", "Yield"),
    ("Production", "Yield")
]

for x, y in relationships:

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x=x,
        y=y,
        alpha=0.4
    )

    plt.title(f"{x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)

    plt.tight_layout()

    filename = f"EDA_Outputs/{x}_vs_{y}.png"

    plt.savefig(filename, dpi=300)

    plt.close()

# ------------------------------------------------------------
# 12. CORRELATION MATRIX
# ------------------------------------------------------------

print("\nCalculating correlations...")

correlation = df[numeric_cols].corr()

correlation.to_csv(
    "EDA_Outputs/correlation_matrix_eda.csv"
)

print("\nCorrelation Matrix:")
print(correlation.round(3))

plt.figure(figsize=(10, 8))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Numerical Variables")

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/correlation_heatmap_eda.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 13. YEAR-WISE TREND ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. YEAR-WISE TREND ANALYSIS")
print("=" * 70)

year_analysis = (
    df.groupby("Crop_Year")
    .agg({
        "Production": "sum",
        "Yield": "mean",
        "Annual_Rainfall": "mean"
    })
    .reset_index()
)

year_analysis.to_csv(
    "EDA_Outputs/year_wise_analysis.csv",
    index=False
)

# Production trend

plt.figure(figsize=(11, 6))

plt.plot(
    year_analysis["Crop_Year"],
    year_analysis["Production"],
    marker="o"
)

plt.title("Year-wise Total Crop Production")
plt.xlabel("Year")
plt.ylabel("Total Production")

plt.grid(True, alpha=0.3)

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
    year_analysis["Yield"],
    marker="o"
)

plt.title("Year-wise Average Crop Yield")
plt.xlabel("Year")
plt.ylabel("Average Yield")

plt.grid(True, alpha=0.3)

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
    year_analysis["Annual_Rainfall"],
    marker="o"
)

plt.title("Year-wise Average Annual Rainfall")
plt.xlabel("Year")
plt.ylabel("Average Rainfall")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "EDA_Outputs/year_wise_rainfall_trend.png",
    dpi=300
)

plt.close()

# ------------------------------------------------------------
# 14. ZERO PRODUCTION ANALYSIS
# ------------------------------------------------------------

zero_production = df[df["Production"] == 0]

zero_production_summary = pd.DataFrame({
    "Metric": [
        "Zero Production Records",
        "Percentage of Dataset"
    ],
    "Value": [
        len(zero_production),
        round((len(zero_production) / len(df)) * 100, 2)
    ]
})

zero_production_summary.to_csv(
    "EDA_Outputs/zero_production_analysis.csv",
    index=False
)

print("\nZero Production Records:", len(zero_production))
print(
    "Percentage:",
    round((len(zero_production) / len(df)) * 100, 2),
    "%"
)

# ------------------------------------------------------------
# 15. HIGH-YIELD ANOMALIES
# ------------------------------------------------------------

yield_q1 = df["Yield"].quantile(0.25)
yield_q3 = df["Yield"].quantile(0.75)

yield_iqr = yield_q3 - yield_q1

yield_upper = yield_q3 + (1.5 * yield_iqr)

high_yield = df[df["Yield"] > yield_upper]

high_yield.to_csv(
    "EDA_Outputs/high_yield_anomalies.csv",
    index=False
)

print("\nHigh-yield statistical anomalies:", len(high_yield))

# ------------------------------------------------------------
# 16. STRONGEST CORRELATIONS
# ------------------------------------------------------------

corr_pairs = correlation.where(
    np.triu(
        np.ones(correlation.shape),
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
    corr_pairs["Correlation"].abs()
)

corr_pairs = corr_pairs.sort_values(
    "Absolute_Correlation",
    ascending=False
)

corr_pairs.to_csv(
    "EDA_Outputs/strongest_correlations.csv",
    index=False
)

print("\nStrongest Correlations:")
print(corr_pairs.head(10))

# ------------------------------------------------------------
# 17. AUTOMATIC FINDINGS FILE
# ------------------------------------------------------------

top_crop = crop_production.index[0]
top_crop_value = crop_production.iloc[0]

top_state = state_production.index[0]
top_state_value = state_production.iloc[0]

best_season = season_yield.index[0]
best_season_yield = season_yield.iloc[0]

strongest = corr_pairs.iloc[0]

findings = [
    f"The dataset contains {len(df):,} observations across {len(df.columns)} variables.",
    f"{df['Crop'].nunique()} different crops and {df['State'].nunique()} states are represented.",
    f"{df['Season'].nunique()} agricultural seasons are represented.",
    f"{top_crop} has the highest total production among the crops analysed, with total production of {top_crop_value:,.2f}.",
    f"{top_state} has the highest total production among the states analysed, with total production of {top_state_value:,.2f}.",
    f"{best_season} has the highest average yield at approximately {best_season_yield:.3f}.",
    f"The strongest numerical correlation in the dataset is between {strongest['Variable_1']} and {strongest['Variable_2']}, with a correlation of {strongest['Correlation']:.3f}.",
    f"There are {len(zero_production):,} records with zero production, representing {len(zero_production)/len(df)*100:.2f}% of the dataset.",
    f"{len(high_yield):,} observations were identified as high-yield statistical anomalies using the IQR-based threshold.",
    "Correlation values describe statistical association and should not automatically be interpreted as proof of causation."
]

with open(
    "EDA_Outputs/eda_findings.txt",
    "w",
    encoding="utf-8"
) as file:

    for i, finding in enumerate(findings, 1):
        file.write(f"{i}. {finding}\n")

# ------------------------------------------------------------
# COMPLETE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETED SUCCESSFULLY!")
print("=" * 70)

print("\nAll outputs saved inside:")
print("EDA_Outputs")

print("\nGenerated:")
print("- Statistical summaries")
print("- Distribution plots")
print("- Boxplots")
print("- Crop analysis")
print("- State analysis")
print("- Season analysis")
print("- Relationship plots")
print("- Correlation heatmap")
print("- Year-wise trends")
print("- Anomaly analysis")
print("- EDA findings")
print("- CSV analysis files")

print("\nWeek 2 EDA is ready.")