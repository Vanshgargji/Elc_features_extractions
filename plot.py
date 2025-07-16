import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load cleaned dataset ===
df = pd.read_csv(r"D:\elc_2nd\driver_cleaned.csv")

# === Auto-extract original class names ===
# Step 1: Load unencoded dataset (before label encoding)
original_df = pd.read_csv(r"D:\elc_2nd\driver_drowsiness_features.csv")

# Step 2: Build mapping from actual class names to encoded labels
unique_labels = original_df['label'].unique()
label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
int_to_label = {v: k for k, v in label_to_int.items()}  # Inverse map

# === Extract features and target from cleaned dataset ===
X = df.drop('label', axis=1)
y = df['label']

# === Split into train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Load trained LightGBM model ===
model = joblib.load(r"D:\elc_2nd\drowsiness_lgbm_model.pkl")

# === Predict test set ===
y_pred = model.predict(X_test)

# === Build class name list in correct order ===
class_names = [int_to_label[i] for i in sorted(df['label'].unique())]

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)

# === Plot Heatmap ===
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Confusion Matrix - Driver Drowsiness Detection")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.tight_layout()
plt.show()

# === Print Classification Report ===
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))


