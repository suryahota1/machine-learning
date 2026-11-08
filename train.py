import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

RANDOM_STATE = 42


# 1. Load Data directly from test_data.csv
df = pd.read_csv('./test_data.csv')

# 2. Separate features and target
X = df.drop(columns=['Class'])
target_raw = df['Class']

le = LabelEncoder()
target = le.fit_transform(target_raw)

# 3. Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=RANDOM_STATE, stratify=target)

# Save test split as test_data.csv for deployment upload requirement
test_df = X_test.copy()
test_df['Class'] = le.inverse_transform(y_test)
test_df.to_csv('deployment_test_data.csv', index=False)
print("Saved deployment_test_data.csv successfully!", test_df.shape)

# 4. Feature Scaling (essential for LR, KNN, SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

os.makedirs('model', exist_ok=True)
joblib.dump(scaler, 'model/scaler.joblib')
joblib.dump(le, 'model/encoder.joblib')


# 5. Train & Serialize Models
models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes": GaussianNB(),
    "svm": SVC(probability=True, random_state=RANDOM_STATE)
}

for name, model in models.items():
    if name in ["logistic_regression", "knn", "svm"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    joblib.dump(model, f'model/{name}.joblib')
    print(f"Trained & Saved: {name}.joblib")





    