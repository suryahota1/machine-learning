import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Dry Bean Classifier Evaluation", layout="wide")

@st.cache_resource
def load_artifacts():
    scaler = joblib.load('model/scaler.joblib')
    encoder = joblib.load('model/encoder.joblib')
    models = {
        "Logistic Regression": joblib.load('model/logistic_regression.joblib'),
        "Decision Tree": joblib.load('model/decision_tree.joblib'),
        "K-Nearest Neighbors": joblib.load('model/knn.joblib'),
        "Naive Bayes": joblib.load('model/naive_bayes.joblib'),
        "Support Vector Machine": joblib.load('model/svm.joblib')
    }
    return scaler, encoder, models

scaler, encoder, models = load_artifacts()

st.title("🌱 Dry Bean Multi-Class Classification Dashboard")
st.markdown("Upload your `test_data.csv` file to evaluate pre-trained machine learning models on morphological dry bean data.")

# 1. Dataset Upload Option
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)
    st.success("Test dataset loaded successfully!")
    
    if 'Class' not in test_df.columns:
        st.error("Uploaded CSV must contain the target column 'Class'.")
    else:
        X_test = test_df.drop(columns=['Class'])
        y_test_raw = test_df['Class']
        y_test = encoder.transform(y_test_raw)
        
        # 2. Model Selection Dropdown
        selected_model_name = st.selectbox("Select Classification Model", list(models.keys()))
        model = models[selected_model_name]
        
        # Preprocessing choice based on model architecture
        if selected_model_name in ["Logistic Regression", "K-Nearest Neighbors", "Support Vector Machine"]:
            X_test_processed = scaler.transform(X_test)
        else:
            X_test_processed = X_test
            
        # Predictions
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)
        
        # 3. Calculation of Evaluation Metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        mcc = matthews_corrcoef(y_test, y_pred)
        
        # Display Metrics Dashboard
        st.subheader(f"Performance Metrics: {selected_model_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.4f}")
        col2.metric("AUC Score", f"{auc:.4f}")
        col3.metric("Precision", f"{prec:.4f}")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", f"{rec:.4f}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC Score", f"{mcc:.4f}")
        
        # 4. Confusion Matrix Visualization
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=encoder.classes_, yticklabels=encoder.classes_, ax=ax)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(fig)
        
        # 5. Classification Report
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())
else:
    st.info("Please upload your `test_data.csv` file to begin evaluation.")







    