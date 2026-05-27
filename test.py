import pandas as pd

# Load your dataset
df = pd.read_csv("final_combined_dataset.csv")

# Replace 'Unknown' or invalid entries with NaN
df.replace("Unknown", pd.NA, inplace=True)

# Drop rows with missing critical info
df.dropna(subset=['state','age_group','gender','risk_label'], inplace=True)

# Optional: Reset index after dropping rows
df.reset_index(drop=True, inplace=True)

# Verify
print(df.head())
print(df.isna().sum())
