import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import shap
import joblib
import matplotlib.pyplot as plt

# Load Data
df = pd.read_csv("your_data.csv")

# Feature selection
structured_cols = ['Tenure', 'MonthlyCharges', 'TotalCharges']  # update if needed
text_col = 'Feedback'
target = 'Churn'

# Preprocessing
df[text_col] = df[text_col].fillna("")
df[structured_cols] = df[structured_cols].fillna(0)
X_structured = df[structured_cols]
X_text = df[text_col]
y = df[target].map({'No': 0, 'Yes': 1})

# Feature engineering
scaler = StandardScaler()
X_structured_scaled = scaler.fit_transform(X_structured)

tfidf = TfidfVectorizer(max_features=100)
X_text_vec = tfidf.fit_transform(X_text).toarray()

X_final = np.hstack((X_structured_scaled, X_text_vec))

# Train model
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_final, y)

# ✅ Save model components
joblib.dump(xgb, 'churn_model.pkl')
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ✅ SHAP analysis
explainer = shap.Explainer(xgb, X_final)
shap_values = explainer(X_final)

# Optional: Save summary plot
shap.summary_plot(shap_values, X_final, show=False)
plt.savefig("shap_summary_plot.png")

# Optional: Show force plot for 1 prediction
# shap.initjs()
# shap.force_plot(explainer.expected_value, shap_values[0], matplotlib=True)
