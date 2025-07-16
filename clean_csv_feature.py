import pandas as pd
from sklearn.preprocessing import LabelEncoder

# === Step 1: Load the dataset ===
CSV_PATH = r"D:\elc_2nd\driver_drowsiness_features.csv"
df = pd.read_csv(CSV_PATH)
print("Original shape:", df.shape)

# === Step 2: Drop duplicate rows ===
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# === Step 3: Drop rows with missing values ===
df = df.dropna()
print("After dropping NaNs:", df.shape)

# === Step 4: Check class distribution ===
print("\nClass distribution before encoding:")
print(df['label'].value_counts())

# === Step 5: Encode labels (categorical to numeric) ===
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save label mapping for reference
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("\nLabel encoding mapping:")
for k, v in label_mapping.items():
    print(f"{k}: {v}")

# === Step 6: Shuffle the dataset ===
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# === Step 7: Save cleaned CSV ===
CLEANED_CSV_PATH = r"D:\elc_2nd\driver_cleaned.csv"
df.to_csv(CLEANED_CSV_PATH, index=False)
print(f"\n✅ Cleaned CSV saved to: {CLEANED_CSV_PATH}")
