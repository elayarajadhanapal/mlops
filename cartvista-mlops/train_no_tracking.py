"""The BEFORE picture: training with no memory of what was done."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
 
df = pd.read_csv("churn_data.csv")
X, y = df.drop(columns=["churn"]), df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
 
# run 1
# model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42) 
# AUC=0.7001  accuracy=0.9064

# run 2
# model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
# AUC=0.7085  accuracy=0.9064

# run 3
# model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
# AUC=0.6961  accuracy=0.9056

# run 4
# model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
# AUC=0.6811  accuracy=0.9064

# run 5
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
# AUC=0.6568  accuracy=0.9080

model.fit(X_tr, y_tr)
 
auc = roc_auc_score(y_te, model.predict_proba(X_te)[:, 1])
acc = accuracy_score(y_te, model.predict(X_te))
print(f"AUC={auc:.4f}  accuracy={acc:.4f}")

