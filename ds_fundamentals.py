"""
Data Science Fundamentals Assessment - companion code
Sections: 1) Python basics  2) Statistical foundations  3) Data types  4) Best practices
Run:  python ds_fundamentals.py
Requires: numpy, pandas, scipy
"""
import numpy as np
import pandas as pd
from scipy import stats

np.random.seed(42)  # reproducibility


def section(title):
    print("\n" + "=" * 8, title, "=" * 8)


# ---------------------------------------------------------------- 1. PYTHON BASICS
section("1. PYTHON BASICS")

scores = [72, 85, 90, 66, 85, 78]                  # list
student = {"name": "Asha", "age": 21, "score": 85}  # dictionary
point = (3, 4)                                      # tuple
unique_scores = set(scores)                         # set

print("List:", scores)
print("Dict:", student)
print("Unique scores:", sorted(unique_scores))

squares = [x ** 2 for x in range(1, 6)]            # list comprehension
print("Squares:", squares)


def grade(score):
    """Convert a numeric score to a letter grade."""
    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    return "C"


print("Grades:", [grade(s) for s in scores])

arr = np.array(scores)
print("NumPy mean/std:", arr.mean().round(2), arr.std(ddof=1).round(2))

# ---------------------------------------------------------------- 2. STATISTICS
section("2. STATISTICAL FOUNDATIONS")

s = pd.Series(scores)
print("Mean:", round(s.mean(), 2))
print("Median:", s.median())
print("Mode:", s.mode().tolist())
print("Variance (sample):", round(s.var(), 2))
print("Std dev (sample):", round(s.std(), 2))
print("Range:", s.max() - s.min())

# Correlation on synthetic data: study hours vs exam score
hours = np.random.uniform(1, 10, 50)
exam = 40 + 5 * hours + np.random.normal(0, 5, 50)
r = np.corrcoef(hours, exam)[0, 1]
print("Correlation (hours vs score):", round(r, 3))

# Hypothesis test: do two teaching methods differ in mean score?
group_a = np.random.normal(70, 8, 40)
group_b = np.random.normal(75, 8, 40)
t_stat, p_val = stats.ttest_ind(group_a, group_b)
print("Two-sample t-test: t =", round(t_stat, 3), ", p =", round(p_val, 4))
print("Reject H0 at 0.05?", p_val < 0.05)

# ---------------------------------------------------------------- 3. DATA TYPES
section("3. DATA TYPES")

df = pd.DataFrame({
    "student_id": [101, 102, 103, 104, 105],               # identifier
    "age": [21, 22, 20, 23, 21],                           # numerical (discrete)
    "gpa": [3.4, 3.8, 2.9, 3.1, 3.6],                      # numerical (continuous)
    "city": ["Mumbai", "Pune", "Mumbai", "Delhi", "Pune"],  # categorical (nominal)
    "level": ["Low", "High", "Medium", "Medium", "High"],   # categorical (ordinal)
})
df["level"] = pd.Categorical(df["level"], categories=["Low", "Medium", "High"], ordered=True)
df["city"] = df["city"].astype("category")
print(df.dtypes)
print("\nCity counts:\n", df["city"].value_counts())
print("\nStudents at level >= Medium:", (df["level"] >= "Medium").sum())

# ---------------------------------------------------------------- 4. BEST PRACTICES
section("4. BEST PRACTICES")

raw = pd.DataFrame({
    "id": [1, 2, 2, 3, 4, 5, 6],
    "salary": [50000, 62000, 62000, None, 58000, 900000, 61000],
    "dept": ["IT", "HR", "HR", "IT", None, "IT", "HR"],
})
print("Raw shape:", raw.shape)
print("Missing values:\n", raw.isna().sum())

clean = raw.drop_duplicates()                                    # remove duplicates
clean = clean.assign(
    salary=clean["salary"].fillna(clean["salary"].median()),     # impute numeric
    dept=clean["dept"].fillna("Unknown"),                        # impute categorical
)

# Outlier detection with the IQR rule
q1, q3 = clean["salary"].quantile([0.25, 0.75])
iqr = q3 - q1
mask = (clean["salary"] < q1 - 1.5 * iqr) | (clean["salary"] > q3 + 1.5 * iqr)
print("\nOutliers flagged:\n", clean[mask])
print("Clean shape:", clean.shape)

# Avoiding data leakage: split first, then compute statistics on TRAIN only
data = pd.DataFrame({"x": np.random.normal(50, 10, 100)})
train = data.sample(frac=0.8, random_state=42)
test = data.drop(train.index)
mu, sd = train["x"].mean(), train["x"].std()     # fit on train only
test_scaled = (test["x"] - mu) / sd              # apply to test
print("\nTrain/test sizes:", len(train), len(test))
print("Test scaled mean (not exactly 0 - that is expected):", round(test_scaled.mean(), 3))
