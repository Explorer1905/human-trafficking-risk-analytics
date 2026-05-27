import os
import pandas as pd
import numpy as np

# ----- Config -----
DATASETS_FOLDER = "datasets"
OUTPUT_FILE = "final_combined_dataset.csv"

# High-risk age group
HIGH_RISK_AGE = "Below 18"

# ----- Helper functions -----
def is_victim_file(filename):
    """Check if a file contains victim data based on keywords."""
    keywords = ["victims", "rescued"]
    return any(k in filename.lower() for k in keywords)

def expand_victim_counts(df, year=None):
    """
    Expand aggregated victim counts to one row per victim.
    Non-numeric counts (e.g., 'na') are treated as 0.
    """
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    expanded_rows = []
    for _, row in df.iterrows():
        state = row.get('state/ut', 'Unknown')
        age_group = row.get('age_group', 'Unknown')
        region_type = row.get('region_type', 'Unknown')

        # Male victims
        male_count = pd.to_numeric(row.get('male', 0), errors='coerce')
        male_count = int(male_count) if not pd.isna(male_count) else 0
        for _ in range(male_count):
            expanded_rows.append({
                "state": state,
                "age_group": age_group,
                "gender": "Male",
                "region_type": region_type,
                "risk_label": "High" if str(age_group).lower() == HIGH_RISK_AGE.lower() else "Low",
                "year": year if year else 'Unknown'
            })

        # Female victims
        female_count = pd.to_numeric(row.get('female', 0), errors='coerce')
        female_count = int(female_count) if not pd.isna(female_count) else 0
        for _ in range(female_count):
            expanded_rows.append({
                "state": state,
                "age_group": age_group,
                "gender": "Female",
                "region_type": region_type,
                "risk_label": "High" if str(age_group).lower() == HIGH_RISK_AGE.lower() else "Low",
                "year": year if year else 'Unknown'
            })

    return pd.DataFrame(expanded_rows)

def extract_year_from_filename(filename):
    """Extract year from filename (e.g., '2018_victims_rescued.xlsx' -> '2018')"""
    import re
    match = re.search(r'(20\d{2})', filename)
    return match.group(1) if match else None

def load_victim_files(folder):
    """Load all Excel victim files and expand them."""
    all_files = [f for f in os.listdir(folder) if f.endswith(".xlsx")]
    combined_list = []
    for file in all_files:
        path = os.path.join(folder, file)
        print(f"Loading {file}...")
        if not is_victim_file(file):
            print(f"Skipping {file}, not a victim dataset")
            continue
        try:
            year = extract_year_from_filename(file)
            df = pd.read_excel(path)
            expanded = expand_victim_counts(df, year)
            combined_list.append(expanded)
            print(f"Expanded {len(expanded)} victim records from {file} (Year: {year})")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    if combined_list:
        combined_df = pd.concat(combined_list, ignore_index=True, sort=False)
        print(f"Total combined victim records: {len(combined_df)}")
        return combined_df
    else:
        print("No victim files found or loaded successfully.")
        return pd.DataFrame()

# ----- Main -----
def main():
    combined = load_victim_files(DATASETS_FOLDER)
    if combined.empty:
        print("No data to save.")
        return

    # Fill missing columns if any
    for col in ['state', 'age_group', 'gender', 'region_type', 'risk_label', 'year']:
        if col not in combined.columns:
            combined[col] = 'Unknown'

    # Save final cleaned dataset
    combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Cleaned dataset saved as {OUTPUT_FILE}")
    print("Risk label counts:")
    print(combined['risk_label'].value_counts())

if __name__ == "__main__":
    main()
