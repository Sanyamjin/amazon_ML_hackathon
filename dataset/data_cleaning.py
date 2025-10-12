import pandas as pd
import re
from tqdm import tqdm

tqdm.pandas()  # Progress bar for pandas apply

# =========================================================
#  Function to Parse the Catalog Content Field
# =========================================================
def parse_catalog_content(text):
    if pd.isna(text):
        return {
            "item_name": None,
            "prod_desc": None,
            "bullet_points": [],
            "value": None,
            "unit": None
        }

    # Extract Item Name
    item_name_match = re.search(r"Item Name:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    item_name = item_name_match.group(1).strip() if item_name_match else None

    # Extract Product Description
    prod_desc_match = re.search(r"Product Description:\s*(.*?)(?:Value:|Unit:|$)", text, re.IGNORECASE | re.DOTALL)
    prod_desc = prod_desc_match.group(1).strip() if prod_desc_match else None

    # Extract Bullet Points (1–5)
    bullet_points = re.findall(r"Bullet Point\s*\d*:\s*(.*)", text, re.IGNORECASE)
    bullet_points = [bp.strip() for bp in bullet_points if bp.strip()]

    # Extract Value
    value_match = re.search(r"Value:\s*([\d\.]+)", text, re.IGNORECASE)
    value = float(value_match.group(1)) if value_match else None

    # Extract Unit
    unit_match = re.search(r"Unit:\s*(.*)", text, re.IGNORECASE)
    unit = unit_match.group(1).strip().lower() if unit_match else None
    if unit:
        unit = unit.replace("fl ", "fl_").replace(" ", "_")

    return {
        "item_name": item_name,
        "prod_desc": prod_desc,
        "bullet_points": bullet_points,
        "value": value,
        "unit": unit
    }

# =========================================================
#  Main Cleaning Function
# =========================================================
def clean_dataset(input_path, output_path, is_test=False):
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows")

    # Parse catalog_content into structured fields
    print("Parsing catalog_content ...")
    parsed = df["catalog_content"].progress_apply(parse_catalog_content).apply(pd.Series)

    # Merge parsed columns back
    df = pd.concat([df, parsed], axis=1)

    # Drop old text column
    df.drop(columns=["catalog_content"], inplace=True)

    # Standardize column names
    df.rename(columns={"sample_id": "prod_id"}, inplace=True)

    # Flatten bullet_points list into string
    df["bullet_points"] = df["bullet_points"].apply(lambda x: "; ".join(x) if isinstance(x, list) else x)

    # Derived features
    df["num_bullets"] = df["bullet_points"].apply(lambda x: len(x.split(";")) if isinstance(x, str) and x.strip() else 0)
    df["has_description"] = df["prod_desc"].apply(lambda x: 1 if isinstance(x, str) and len(x.strip()) > 0 else 0)

    # Only compute price_per_unit if price column exists and not test
    if not is_test and "price" in df.columns:
        df["price_per_unit"] = df.apply(
            lambda row: row["price"] / row["value"] if pd.notnull(row["price"]) and pd.notnull(row["value"]) and row["value"] > 0 else None,
            axis=1
        )

    # Clean item names
    df["item_name"] = df["item_name"].apply(lambda x: re.sub(r"\s+", " ", x.strip()) if isinstance(x, str) else x)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path}")
    print("\nColumns available:", list(df.columns))
    print(df.head(3))

# =========================================================
#  Entry Point
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean and parse product catalog dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV (train.csv or test.csv)")
    parser.add_argument("--output", type=str, default="cleaned_output.csv", help="Path to save cleaned CSV")
    parser.add_argument("--is_test", action="store_true", help="Set this flag if input is a test set (no price column)")
    args = parser.parse_args()

    clean_dataset(args.input, args.output, is_test=args.is_test)
