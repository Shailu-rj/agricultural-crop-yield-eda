import pandas as pd
import numpy as np

# ============================================================
# MACHINE LEARNING DATA ANALYST INTERNSHIP - TASK 1
# AGRIBUSINESS DATA COLLECTION AND INITIAL ANALYSIS
# ============================================================

# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("crop_yield.csv")

print("\n" + "=" * 70)
print("AGRIBUSINESS DATASET - INITIAL ANALYSIS")
print("=" * 70)


# 2. DATASET DIMENSIONS
# ------------------------------------------------------------

print("\n[1] DATASET DIMENSIONS")
print("-" * 70)

print("Number of Rows    :", df.shape[0])
print("Number of Columns :", df.shape[1])


# 3. COLUMN NAMES
# ------------------------------------------------------------

print("\n[2] COLUMN NAMES")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")


# 4. FIRST FIVE RECORDS
# ------------------------------------------------------------

print("\n[3] FIRST 5 RECORDS")
print("-" * 70)

print(df.head())


# 5. LAST FIVE RECORDS
# ------------------------------------------------------------

print("\n[4] LAST 5 RECORDS")
print("-" * 70)

print(df.tail())


# 6. DATA TYPES
# ------------------------------------------------------------

print("\n[5] DATA TYPES")
print("-" * 70)

print(df.dtypes)


# 7. DATASET INFORMATION
# ------------------------------------------------------------

print("\n[6] DATASET INFORMATION")
print("-" * 70)

df.info()


# 8. MISSING VALUES
# ------------------------------------------------------------

print("\n[7] MISSING VALUES")
print("-" * 70)

missing_values = df.isnull().sum()

print(missing_values)

print("\nTotal Missing Values:", df.isnull().sum().sum())


# 9. DUPLICATE RECORDS
# ------------------------------------------------------------

print("\n[8] DUPLICATE RECORDS")
print("-" * 70)

duplicate_count = df.duplicated().sum()

print("Total Duplicate Rows:", duplicate_count)


# 10. STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n[9] STATISTICAL SUMMARY")
print("-" * 70)

print(df.describe())


# 11. CATEGORICAL DATA ANALYSIS
# ------------------------------------------------------------

print("\n[10] CATEGORICAL DATA ANALYSIS")
print("-" * 70)

print("Number of Crops   :", df["Crop"].nunique())
print("Number of States  :", df["State"].nunique())
print("Number of Seasons :", df["Season"].nunique())


print("\nCrop Categories:")
print(df["Crop"].unique())

print("\nStates:")
print(df["State"].unique())

print("\nSeasons:")
print(df["Season"].unique())


# 12. YEAR ANALYSIS
# ------------------------------------------------------------

print("\n[11] YEAR ANALYSIS")
print("-" * 70)

print("Minimum Crop Year:", df["Crop_Year"].min())
print("Maximum Crop Year:", df["Crop_Year"].max())


# 13. NUMERICAL COLUMN ANALYSIS
# ------------------------------------------------------------

print("\n[12] NUMERICAL COLUMN ANALYSIS")
print("-" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:

    print("\nColumn:", column)
    print("Minimum :", df[column].min())
    print("Maximum :", df[column].max())
    print("Mean    :", df[column].mean())
    print("Median  :", df[column].median())
    print("Std Dev :", df[column].std())


# 14. CHECK FOR NEGATIVE VALUES
# ------------------------------------------------------------

print("\n[13] NEGATIVE VALUE CHECK")
print("-" * 70)

for column in numeric_columns:

    negative_count = (df[column] < 0).sum()

    print(f"{column}: {negative_count} negative values")


# 15. UNIQUE VALUE COUNTS
# ------------------------------------------------------------

print("\n[14] UNIQUE VALUE COUNT PER COLUMN")
print("-" * 70)

for column in df.columns:
    print(f"{column}: {df[column].nunique()} unique values")


# 16. MOST COMMON CROPS
# ------------------------------------------------------------

print("\n[15] TOP 10 MOST FREQUENT CROPS")
print("-" * 70)

print(df["Crop"].value_counts().head(10))


# 17. MOST COMMON STATES
# ------------------------------------------------------------

print("\n[16] TOP 10 STATES BY NUMBER OF RECORDS")
print("-" * 70)

print(df["State"].value_counts().head(10))


# 18. BASIC DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n[17] DATA QUALITY CHECK")
print("-" * 70)

print("Missing Values :", df.isnull().sum().sum())
print("Duplicate Rows :", df.duplicated().sum())

print("\nData quality check completed.")


# 19. SAVE INITIAL COPY
# ------------------------------------------------------------

df.to_csv("initial_crop_yield_dataset.csv", index=False)

print("\nInitial dataset copy saved as:")
print("initial_crop_yield_dataset.csv")


# ============================================================
# END OF INITIAL ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("INITIAL DATA ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 70)

# ============================================================
# STEP 2 - DATA QUALITY AND OUTLIER DETECTION
# ============================================================

print("\n" + "=" * 70)
print("STEP 2 - DATA QUALITY AND OUTLIER DETECTION")
print("=" * 70)


# ------------------------------------------------------------
# 1. CHECK DATA TYPES
# ------------------------------------------------------------

print("\n[1] CHECKING DATA TYPES")
print("-" * 70)

print(df.dtypes)


# ------------------------------------------------------------
# 2. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\n[2] CHECKING MISSING VALUES")
print("-" * 70)

missing = df.isnull().sum()

print(missing)

if missing.sum() == 0:
    print("\nResult: No missing values found.")
else:
    print("\nResult: Missing values detected.")


# ------------------------------------------------------------
# 3. CHECK DUPLICATES
# ------------------------------------------------------------

print("\n[3] CHECKING DUPLICATE RECORDS")
print("-" * 70)

duplicates = df.duplicated().sum()

print("Duplicate rows:", duplicates)

if duplicates == 0:
    print("Result: No duplicate records found.")
else:
    print("Result: Duplicate records detected.")


# ------------------------------------------------------------
# 4. CHECK NEGATIVE VALUES
# ------------------------------------------------------------

print("\n[4] CHECKING NEGATIVE VALUES")
print("-" * 70)

numeric_columns = df.select_dtypes(include=np.number).columns

for column in numeric_columns:
    negative_count = (df[column] < 0).sum()
    print(f"{column}: {negative_count} negative values")


# ------------------------------------------------------------
# 5. CHECK ZERO VALUES
# ------------------------------------------------------------

print("\n[5] CHECKING ZERO VALUES")
print("-" * 70)

for column in numeric_columns:
    zero_count = (df[column] == 0).sum()
    print(f"{column}: {zero_count} zero values")


# ------------------------------------------------------------
# 6. OUTLIER DETECTION USING IQR
# ------------------------------------------------------------

print("\n[6] OUTLIER DETECTION USING IQR METHOD")
print("-" * 70)

outlier_summary = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    outlier_count = len(outliers)

    outlier_summary.append({
        "Column": column,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Outlier Count": outlier_count
    })

    print(f"\nColumn: {column}")
    print(f"Q1           : {Q1:.4f}")
    print(f"Q3           : {Q3:.4f}")
    print(f"IQR          : {IQR:.4f}")
    print(f"Lower Bound  : {lower_bound:.4f}")
    print(f"Upper Bound  : {upper_bound:.4f}")
    print(f"Outliers     : {outlier_count}")


# ------------------------------------------------------------
# 7. CREATE OUTLIER SUMMARY TABLE
# ------------------------------------------------------------

outlier_df = pd.DataFrame(outlier_summary)

print("\n" + "=" * 70)
print("OUTLIER SUMMARY TABLE")
print("=" * 70)

print(outlier_df.to_string(index=False))


# ------------------------------------------------------------
# 8. CHECK CATEGORICAL CONSISTENCY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL DATA CONSISTENCY")
print("=" * 70)


print("\nSeason values:")
print(df["Season"].value_counts())


print("\nCrop values:")
print(df["Crop"].value_counts().head(20))


print("\nState values:")
print(df["State"].value_counts())


# ------------------------------------------------------------
# 9. CHECK FOR EXTRA SPACES IN TEXT DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING TEXT FORMATTING")
print("=" * 70)

text_columns = df.select_dtypes(include="object").columns

for column in text_columns:

    space_count = df[column].astype(str).str.strip().ne(
        df[column].astype(str)
    ).sum()

    print(f"{column}: {space_count} values with leading/trailing spaces")


# ------------------------------------------------------------
# 10. SAVE OUTLIER REPORT
# ------------------------------------------------------------

outlier_df.to_csv("outlier_analysis.csv", index=False)

print("\nOutlier analysis saved as:")
print("outlier_analysis.csv")


# ------------------------------------------------------------
# END STEP 2
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 2 COMPLETED")
print("=" * 70)

# ============================================================
# STEP 3 - DATA CLEANING AND VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 3 - DATA CLEANING AND VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. CREATE A COPY OF THE DATASET
# ------------------------------------------------------------

cleaned_df = df.copy()

print("\n[1] Created working copy of original dataset.")


# ------------------------------------------------------------
# 2. REMOVE LEADING/TRAILING SPACES FROM TEXT COLUMNS
# ------------------------------------------------------------

print("\n[2] CLEANING TEXT COLUMNS")
print("-" * 70)

text_columns = cleaned_df.select_dtypes(include="object").columns

for column in text_columns:
    cleaned_df[column] = cleaned_df[column].str.strip()

print("Leading/trailing spaces removed from:")
print(", ".join(text_columns))


# ------------------------------------------------------------
# 3. CHECK TEXT VALUES AFTER CLEANING
# ------------------------------------------------------------

print("\n[3] TEXT CONSISTENCY AFTER CLEANING")
print("-" * 70)

for column in text_columns:

    space_count = cleaned_df[column].str.strip().ne(
        cleaned_df[column]
    ).sum()

    print(f"{column}: {space_count} values with extra spaces")


# ------------------------------------------------------------
# 4. CHECK ZERO PRODUCTION RECORDS
# ------------------------------------------------------------

print("\n[4] ZERO PRODUCTION ANALYSIS")
print("-" * 70)

zero_production = cleaned_df[
    cleaned_df["Production"] == 0
]

print("Rows with zero production:", len(zero_production))

if len(zero_production) > 0:
    print("\nSample zero-production records:")
    print(
        zero_production[
            [
                "Crop",
                "Crop_Year",
                "Season",
                "State",
                "Area",
                "Production",
                "Yield"
            ]
        ].head(10).to_string(index=False)
    )


# ------------------------------------------------------------
# 5. CHECK ZERO YIELD RECORDS
# ------------------------------------------------------------

print("\n[5] ZERO YIELD ANALYSIS")
print("-" * 70)

zero_yield = cleaned_df[
    cleaned_df["Yield"] == 0
]

print("Rows with zero yield:", len(zero_yield))

if len(zero_yield) > 0:
    print("\nSample zero-yield records:")
    print(
        zero_yield[
            [
                "Crop",
                "Crop_Year",
                "Season",
                "State",
                "Area",
                "Production",
                "Yield"
            ]
        ].head(10).to_string(index=False)
    )


# ------------------------------------------------------------
# 6. CHECK WHETHER ZERO PRODUCTION AND ZERO YIELD MATCH
# ------------------------------------------------------------

print("\n[6] ZERO PRODUCTION vs ZERO YIELD")
print("-" * 70)

both_zero = cleaned_df[
    (cleaned_df["Production"] == 0) &
    (cleaned_df["Yield"] == 0)
]

print("Rows where both Production and Yield are zero:",
      len(both_zero))


# ------------------------------------------------------------
# 7. CHECK INVALID AREA VALUES
# ------------------------------------------------------------

print("\n[7] AREA VALIDATION")
print("-" * 70)

invalid_area = cleaned_df[
    cleaned_df["Area"] <= 0
]

print("Rows with Area <= 0:", len(invalid_area))


# ------------------------------------------------------------
# 8. CHECK INVALID RAINFALL VALUES
# ------------------------------------------------------------

print("\n[8] RAINFALL VALIDATION")
print("-" * 70)

invalid_rainfall = cleaned_df[
    cleaned_df["Annual_Rainfall"] <= 0
]

print("Rows with Rainfall <= 0:", len(invalid_rainfall))


# ------------------------------------------------------------
# 9. CHECK INVALID FERTILIZER VALUES
# ------------------------------------------------------------

print("\n[9] FERTILIZER VALIDATION")
print("-" * 70)

invalid_fertilizer = cleaned_df[
    cleaned_df["Fertilizer"] <= 0
]

print("Rows with Fertilizer <= 0:", len(invalid_fertilizer))


# ------------------------------------------------------------
# 10. CHECK INVALID PESTICIDE VALUES
# ------------------------------------------------------------

print("\n[10] PESTICIDE VALIDATION")
print("-" * 70)

invalid_pesticide = cleaned_df[
    cleaned_df["Pesticide"] <= 0
]

print("Rows with Pesticide <= 0:", len(invalid_pesticide))


# ------------------------------------------------------------
# 11. CHECK INVALID CROP YEARS
# ------------------------------------------------------------

print("\n[11] CROP YEAR VALIDATION")
print("-" * 70)

invalid_year = cleaned_df[
    (cleaned_df["Crop_Year"] < 1990) |
    (cleaned_df["Crop_Year"] > 2025)
]

print("Rows with suspicious Crop Year:", len(invalid_year))


# ------------------------------------------------------------
# 12. DUPLICATE CHECK AFTER TEXT CLEANING
# ------------------------------------------------------------

print("\n[12] DUPLICATE CHECK AFTER CLEANING")
print("-" * 70)

duplicates_after_cleaning = cleaned_df.duplicated().sum()

print(
    "Duplicate rows after text standardization:",
    duplicates_after_cleaning
)


# ------------------------------------------------------------
# 13. FINAL DATASET SIZE BEFORE OUTLIER TREATMENT
# ------------------------------------------------------------

print("\n[13] DATASET SIZE")
print("-" * 70)

print("Original rows :", len(df))
print("Current rows  :", len(cleaned_df))

print(
    "\nNo observations have been removed based solely on IQR "
    "outlier detection at this stage."
)


# ------------------------------------------------------------
# 14. SAVE CLEANED DATASET - VERSION 1
# ------------------------------------------------------------

cleaned_df.to_csv(
    "cleaned_crop_yield_v1.csv",
    index=False
)

print("\nCleaned dataset saved as:")
print("cleaned_crop_yield_v1.csv")


# ------------------------------------------------------------
# END STEP 3
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 3 COMPLETED")
print("=" * 70)

# ============================================================
# STEP 4 - OUTLIER INVESTIGATION AND DATA VISUALIZATION
# ============================================================

import matplotlib.pyplot as plt
import seaborn as sns

print("\n" + "=" * 70)
print("STEP 4 - OUTLIER INVESTIGATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. INVESTIGATE ZERO YIELD WITH POSITIVE PRODUCTION
# ------------------------------------------------------------

print("\n[1] ZERO YIELD WITH POSITIVE PRODUCTION")
print("-" * 70)

suspicious_yield = cleaned_df[
    (cleaned_df["Yield"] == 0) &
    (cleaned_df["Production"] > 0)
]

print("Number of suspicious records:", len(suspicious_yield))

if len(suspicious_yield) > 0:
    print("\nRecords:")
    print(
        suspicious_yield[
            [
                "Crop",
                "Crop_Year",
                "Season",
                "State",
                "Area",
                "Production",
                "Yield"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 2. CHECK RELATIONSHIP BETWEEN PRODUCTION AND AREA
# ------------------------------------------------------------

print("\n[2] PRODUCTION AND AREA RELATIONSHIP")
print("-" * 70)

positive_area_zero_production = cleaned_df[
    (cleaned_df["Area"] > 0) &
    (cleaned_df["Production"] == 0)
]

print(
    "Records with positive cultivated area but zero production:",
    len(positive_area_zero_production)
)


# ------------------------------------------------------------
# 3. OUTLIER COUNTS USING IQR
# ------------------------------------------------------------

print("\n[3] OUTLIER COUNTS")
print("-" * 70)

outlier_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

outlier_counts = {}

for column in outlier_columns:

    Q1 = cleaned_df[column].quantile(0.25)
    Q3 = cleaned_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (cleaned_df[column] < lower) |
        (cleaned_df[column] > upper)
    ).sum()

    outlier_counts[column] = count

    print(f"{column}: {count}")


# ------------------------------------------------------------
# 4. DISPLAY EXTREME VALUES
# ------------------------------------------------------------

print("\n[4] EXTREME VALUES")
print("-" * 70)

for column in outlier_columns:

    print(f"\nTop 5 highest values - {column}")

    print(
        cleaned_df[
            ["Crop", "State", "Crop_Year", column]
        ]
        .sort_values(by=column, ascending=False)
        .head(5)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# 5. BOXPLOTS FOR NUMERICAL VARIABLES
# ------------------------------------------------------------

print("\n[5] GENERATING BOXPLOTS")
print("-" * 70)

for column in outlier_columns:

    plt.figure(figsize=(10, 5))

    sns.boxplot(x=cleaned_df[column])

    plt.title(f"Boxplot of {column}")
    plt.xlabel(column)

    plt.tight_layout()

    filename = f"boxplot_{column}.png"

    plt.savefig(filename, dpi=300)

    plt.close()

    print(f"Saved: {filename}")


# ------------------------------------------------------------
# 6. HISTOGRAMS
# ------------------------------------------------------------

print("\n[6] GENERATING HISTOGRAMS")
print("-" * 70)

for column in outlier_columns:

    plt.figure(figsize=(10, 5))

    sns.histplot(
        cleaned_df[column],
        kde=True
    )

    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")

    plt.tight_layout()

    filename = f"histogram_{column}.png"

    plt.savefig(filename, dpi=300)

    plt.close()

    print(f"Saved: {filename}")


# ------------------------------------------------------------
# 7. SAVE SUSPICIOUS RECORDS
# ------------------------------------------------------------

suspicious_yield.to_csv(
    "suspicious_zero_yield_records.csv",
    index=False
)

print("\nSuspicious records saved as:")
print("suspicious_zero_yield_records.csv")


# ------------------------------------------------------------
# 8. SAVE OUTLIER COUNTS
# ------------------------------------------------------------

outlier_count_df = pd.DataFrame(
    list(outlier_counts.items()),
    columns=["Column", "Outlier_Count"]
)

outlier_count_df.to_csv(
    "outlier_counts.csv",
    index=False
)

print("Outlier counts saved as:")
print("outlier_counts.csv")


# ------------------------------------------------------------
# END STEP 4
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 4 COMPLETED")
print("=" * 70)

# ============================================================
# STEP 5 - FINAL DATA CLEANING AND NORMALIZATION
# ============================================================

from sklearn.preprocessing import MinMaxScaler

print("\n" + "=" * 70)
print("STEP 5 - FINAL DATA CLEANING AND NORMALIZATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. CREATE FINAL CLEANING COPY
# ------------------------------------------------------------

final_df = df.copy()

print("\n[1] Created final cleaning copy.")


# ------------------------------------------------------------
# 2. STANDARDIZE TEXT COLUMNS
# ------------------------------------------------------------

print("\n[2] STANDARDIZING TEXT DATA")
print("-" * 70)

text_columns = ["Crop", "Season", "State"]

for column in text_columns:
    final_df[column] = final_df[column].astype(str).str.strip()

print("Text standardization completed.")


# ------------------------------------------------------------
# 3. REMOVE DUPLICATES
# ------------------------------------------------------------

print("\n[3] REMOVING DUPLICATES")
print("-" * 70)

before_duplicates = len(final_df)

final_df = final_df.drop_duplicates()

after_duplicates = len(final_df)

print("Rows before duplicate removal:", before_duplicates)
print("Rows after duplicate removal :", after_duplicates)
print("Duplicates removed           :", before_duplicates - after_duplicates)


# ------------------------------------------------------------
# 4. REMOVE IMPOSSIBLE NUMERICAL VALUES
# ------------------------------------------------------------

print("\n[4] VALIDATING NUMERICAL VALUES")
print("-" * 70)

numeric_validation_columns = [
    "Area",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

invalid_rows = pd.Series(False, index=final_df.index)

for column in numeric_validation_columns:

    invalid = final_df[column] <= 0

    print(f"{column} invalid values:", invalid.sum())

    invalid_rows = invalid_rows | invalid


print("Total rows with invalid numerical values:",
      invalid_rows.sum())


# Remove only genuinely impossible values
final_df = final_df[~invalid_rows].copy()


# ------------------------------------------------------------
# 5. HANDLE SUSPICIOUS ZERO YIELD VALUES
# ------------------------------------------------------------

print("\n[5] HANDLING SUSPICIOUS ZERO-YIELD VALUES")
print("-" * 70)

suspicious = final_df[
    (final_df["Yield"] == 0) &
    (final_df["Production"] > 0)
]

print(
    "Records with positive production but zero yield:",
    len(suspicious)
)

if len(suspicious) > 0:

    print("\nThese records will be treated as invalid yield observations.")

    final_df = final_df.drop(suspicious.index)

else:

    print("No suspicious zero-yield records found.")


# ------------------------------------------------------------
# 6. CHECK ZERO PRODUCTION RECORDS
# ------------------------------------------------------------

print("\n[6] ZERO PRODUCTION RECORDS")
print("-" * 70)

zero_production_count = (
    final_df["Production"] == 0
).sum()

print("Zero-production records:", zero_production_count)

print(
    "Zero-production observations are retained because zero "
    "production can represent a genuine agricultural outcome."
)


# ------------------------------------------------------------
# 7. OUTLIER TREATMENT USING WINSORIZATION
# ------------------------------------------------------------

print("\n[7] OUTLIER TREATMENT")
print("-" * 70)

outlier_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

outlier_before = {}
outlier_after = {}

for column in outlier_columns:

    # Calculate IQR
    Q1 = final_df[column].quantile(0.25)
    Q3 = final_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Count outliers before treatment
    before_count = (
        (final_df[column] < lower_bound) |
        (final_df[column] > upper_bound)
    ).sum()

    outlier_before[column] = before_count

    # Winsorization
    final_df[column] = final_df[column].clip(
        lower=lower_bound,
        upper=upper_bound
    )

    # Count outliers after treatment
    after_count = (
        (final_df[column] < lower_bound) |
        (final_df[column] > upper_bound)
    ).sum()

    outlier_after[column] = after_count

    print(f"\n{column}")
    print("Outliers before treatment:", before_count)
    print("Outliers after treatment :", after_count)
    print("Lower bound:", lower_bound)
    print("Upper bound:", upper_bound)


# ------------------------------------------------------------
# 8. SAVE CLEANED DATASET
# ------------------------------------------------------------

final_df.to_csv(
    "cleaned_crop_yield_final.csv",
    index=False
)

print("\n[8] FINAL CLEANED DATASET SAVED")
print("-" * 70)

print("File: cleaned_crop_yield_final.csv")
print("Rows:", final_df.shape[0])
print("Columns:", final_df.shape[1])


# ------------------------------------------------------------
# 9. NORMALIZATION
# ------------------------------------------------------------

print("\n[9] MIN-MAX NORMALIZATION")
print("-" * 70)

normalized_df = final_df.copy()

scaler = MinMaxScaler()

normalized_df[outlier_columns] = scaler.fit_transform(
    normalized_df[outlier_columns]
)

print("Min-Max normalization completed.")

print("\nNormalized numerical columns:")
print(outlier_columns)


# ------------------------------------------------------------
# 10. SAVE NORMALIZED DATASET
# ------------------------------------------------------------

normalized_df.to_csv(
    "normalized_crop_yield_final.csv",
    index=False
)

print("\nNormalized dataset saved as:")
print("normalized_crop_yield_final.csv")


# ------------------------------------------------------------
# 11. FINAL STATISTICAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL STATISTICAL SUMMARY")
print("=" * 70)

print(final_df.describe())


# ------------------------------------------------------------
# 12. BEFORE VS AFTER ROW COUNT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BEFORE VS AFTER CLEANING")
print("=" * 70)

print("Original rows :", len(df))
print("Final rows    :", len(final_df))
print("Rows removed  :", len(df) - len(final_df))


# ------------------------------------------------------------
# 13. FINAL DATA QUALITY CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)

print("Missing values :", final_df.isnull().sum().sum())
print("Duplicate rows :", final_df.duplicated().sum())

print("\nFinal data types:")
print(final_df.dtypes)


# ------------------------------------------------------------
# END STEP 5
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

# ============================================================
# STEP 6 - STATISTICAL ANALYSIS AND VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 - STATISTICAL ANALYSIS AND VISUALIZATION")
print("=" * 70)


# ------------------------------------------------------------
# 1. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------

print("\n[1] DESCRIPTIVE STATISTICS")
print("-" * 70)

statistics = final_df[
    [
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield"
    ]
].describe().T

statistics["median"] = final_df[
    [
        "Area",
        "Production",
        "Annual_Rainfall",
        "Fertilizer",
        "Pesticide",
        "Yield"
    ]
].median()

statistics = statistics[
    [
        "count",
        "mean",
        "median",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max"
    ]
]

print(statistics)


# ------------------------------------------------------------
# 2. CROP-WISE PRODUCTION
# ------------------------------------------------------------

print("\n[2] TOP 10 CROPS BY TOTAL PRODUCTION")
print("-" * 70)

crop_production = (
    final_df.groupby("Crop")["Production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(crop_production)


# ------------------------------------------------------------
# 3. STATE-WISE PRODUCTION
# ------------------------------------------------------------

print("\n[3] TOP 10 STATES BY TOTAL PRODUCTION")
print("-" * 70)

state_production = (
    final_df.groupby("State")["Production"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(state_production)


# ------------------------------------------------------------
# 4. AVERAGE YIELD BY SEASON
# ------------------------------------------------------------

print("\n[4] AVERAGE YIELD BY SEASON")
print("-" * 70)

season_yield = (
    final_df.groupby("Season")["Yield"]
    .mean()
    .sort_values(ascending=False)
)

print(season_yield)


# ------------------------------------------------------------
# 5. AVERAGE RAINFALL BY SEASON
# ------------------------------------------------------------

print("\n[5] AVERAGE RAINFALL BY SEASON")
print("-" * 70)

season_rainfall = (
    final_df.groupby("Season")["Annual_Rainfall"]
    .mean()
    .sort_values(ascending=False)
)

print(season_rainfall)


# ------------------------------------------------------------
# 6. CORRELATION MATRIX
# ------------------------------------------------------------

print("\n[6] CORRELATION MATRIX")
print("-" * 70)

correlation_columns = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

correlation_matrix = final_df[
    correlation_columns
].corr()

print(correlation_matrix.round(3))


# ------------------------------------------------------------
# 7. SAVE STATISTICAL SUMMARY
# ------------------------------------------------------------

statistics.to_csv(
    "statistical_summary.csv"
)

print("\nStatistical summary saved as:")
print("statistical_summary.csv")


# ------------------------------------------------------------
# 8. SAVE CORRELATION MATRIX
# ------------------------------------------------------------

correlation_matrix.to_csv(
    "correlation_matrix.csv"
)

print("Correlation matrix saved as:")
print("correlation_matrix.csv")


# ============================================================
# VISUALIZATIONS
# ============================================================


# ------------------------------------------------------------
# 9. TOP CROPS BY PRODUCTION
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

crop_production.sort_values().plot(kind="barh")

plt.title("Top 10 Crops by Total Production")
plt.xlabel("Total Production")
plt.ylabel("Crop")

plt.tight_layout()

plt.savefig(
    "top_10_crops_production.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 10. TOP STATES BY PRODUCTION
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

state_production.sort_values().plot(kind="barh")

plt.title("Top 10 States by Total Production")
plt.xlabel("Total Production")
plt.ylabel("State")

plt.tight_layout()

plt.savefig(
    "top_10_states_production.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 11. AVERAGE YIELD BY SEASON
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

season_yield.plot(kind="bar")

plt.title("Average Crop Yield by Season")
plt.xlabel("Season")
plt.ylabel("Average Yield")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "average_yield_by_season.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 12. CORRELATION HEATMAP
# ------------------------------------------------------------

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix of Agricultural Variables")

plt.tight_layout()

plt.savefig(
    "correlation_heatmap.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 13. PRODUCTION DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    final_df["Production"],
    kde=True
)

plt.title("Distribution of Crop Production")
plt.xlabel("Production")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "production_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 14. YIELD DISTRIBUTION
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    final_df["Yield"],
    kde=True
)

plt.title("Distribution of Crop Yield")
plt.xlabel("Yield")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "yield_distribution.png",
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 15. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VISUALIZATIONS GENERATED SUCCESSFULLY")
print("=" * 70)

print("""
Generated files:

1. top_10_crops_production.png
2. top_10_states_production.png
3. average_yield_by_season.png
4. correlation_heatmap.png
5. production_distribution.png
6. yield_distribution.png
""")

print("=" * 70)
print("STEP 6 COMPLETED")
print("=" * 70)