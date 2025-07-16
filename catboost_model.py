import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# === Load cleaned dataset ===
df = pd.read_csv(r"D:\elc_2nd\driver_cleaned.csv")

# === Extract features and labels ===
X = df.drop('label', axis=1)
y = df['label']

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Initialize CatBoostClassifier ===
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.05,
    depth=8,
    eval_metric='Accuracy',
    verbose=100,
    task_type='CPU',  # CPU training
    random_seed=42
)

# === Train the model ===
print("🔄 Training CatBoost model...")
model.fit(X_train, y_train)
print("✅ Training complete!")

# === Predict & Evaluate ===
y_pred = model.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)

# === Load class names from original dataset ===
original_df = pd.read_csv(r"D:\elc_2nd\driver_drowsiness_features.csv")
unique_labels = original_df['label'].unique()
label_to_int = {label: idx for idx, label in enumerate(sorted(unique_labels))}
int_to_label = {v: k for k, v in label_to_int.items()}
class_names = [int_to_label[i] for i in sorted(df['label'].unique())]

plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("CatBoost Confusion Matrix - Driver Drowsiness Detection")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# === Save the model ===
joblib.dump(model, r"D:\elc_2nd\drowsiness_catboost_model.pkl")
print("📦 Model saved as 'drowsiness_catboost_model.pkl'")
