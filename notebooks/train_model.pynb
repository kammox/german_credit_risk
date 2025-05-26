import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
import os

# Load data
df = pd.read_csv("notebooks/data/german_credit_data2.csv")

# Drop index column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# Separate features and target
X = df.drop(columns=["Risk"])
y = df["Risk"].map({'good': 0, 'bad': 1})  # 0 = good, 1 = bad (risky)

# Define categorical and numerical columns
categorical_cols = ["Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
numerical_cols = ["Age", "Credit amount", "Duration"]

# Preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("cat", categorical_transformer, categorical_cols),
    ("num", numerical_transformer, numerical_cols)
])

# Final pipeline with model
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))

])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Display evaluation metrics
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

# save model to model/ folder
joblib.dump(model, "notebooks/credit_risk_model.pkl")
print("✅ Modèle sauvegardé dans 'notebooks/credit_risk_model.pkl'")
