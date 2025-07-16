import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load cleaned CSV ===
df = pd.read_csv(r"D:\elc_2nd\driver_cleaned.csv")

# === Features & Labels ===
X = df.drop('label', axis=1)
y = df['label']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# === Initialize and Train XGBoost Model ===
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=1
)

print("🔄 Training XGBoost model...")
model.fit(X_train, y_train)
print("✅ Training complete!")

# === Prediction & Evaluation ===
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)

# Auto-recover class names from original data
original_df = pd.read_csv(r"D:\elc_2nd\driver_drowsiness_features.csv")
unique_labels = original_df['label'].unique()
label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
int_to_label = {v: k for k, v in label_to_int.items()}
class_names = [int_to_label[i] for i in sorted(df['label'].unique())]

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("XGBoost Confusion Matrix - Driver Drowsiness")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === Save the model ===
joblib.dump(model, r"D:\elc_2nd\drowsiness_xgb_model.pkl")
print("📦 Model saved as 'drowsiness_xgb_model.pkl'")
