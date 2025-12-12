import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("heart.csv")
print("Dataset shape:", df.shape)
print(df.head(), "\n")

# Split features and target
X = df.drop("output", axis=1)
y = df["output"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#----------------------------------#
# 1. Random Forest - baseline
#----------------------------------#
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

#----------------------------------#
# 2. Logistic Regression
#----------------------------------#
log = LogisticRegression(max_iter=200)
log.fit(X_train_scaled, y_train)
y_pred_log = log.predict(X_test_scaled)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

#----------------------------------#
# 3. XGBoost
#----------------------------------#
xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                    subsample=0.9, colsample_bytree=0.9, random_state=42)
xgb.fit(X_train_scaled, y_train)
y_pred_xgb = xgb.predict(X_test_scaled)
print("\nXGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

#----------------------------------#
# 4. Tuned Random Forest
#----------------------------------#
tuned_rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=6,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)
tuned_rf.fit(X_train_scaled, y_train)
y_pred_tuned_rf = tuned_rf.predict(X_test_scaled)
print("\nTuned RandomForest Accuracy:", accuracy_score(y_test, y_pred_tuned_rf))
print(classification_report(y_test, y_pred_tuned_rf))

# Feature Importance
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': tuned_rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)
