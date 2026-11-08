# Dry Bean Classification & ML Evaluation Pipeline

## a. Problem Statement
The objective of this project is to build, evaluate, and deploy a robust multi-class classification pipeline to accurately categorize dry bean varieties based on their morphological and geometric features. Accurate automated classification of agricultural grains improves crop sorting efficiency, quality control, and yield assessment.

## b. & c. Dataset Description
- **Dataset Source:** Dry Bean Dataset (UCI Machine Learning Repository)
- **Total Instances:** 13,611 total instances across 7 distinct bean varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, and Sira).
- **Features:** 16 continuous numeric features extracted from bean contours (e.g., Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, Extent, Solidity, Roundness, Compactness, ShapeFactor1–4).
- **Target Variable:** `Class` (7-class multi-class classification).

## GitHub Repository Link
[https://github.com/your-username/dry_beans_classifier](https://github.com/your-username/dry_beans_classifier)

---

## d. Models Used & Evaluation Comparison

### Model Performance Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| **Decision Tree** | 0.8917 | 0.9329 | 0.8913 | 0.8917 | 0.8913 | 0.8690 |
| **KNN** | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| **Naive Bayes** | 0.7639 | 0.9644 | 0.7654 | 0.7639 | 0.7615 | 0.7154 |
| **Support Vector Machine** | **0.9221** | **0.9937** | **0.9224** | **0.9221** | **0.9222** | **0.9059** |

---

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved strong linear separation performance (92.07% accuracy). Bombay beans were classified with absolute perfection, while minor misclassifications occurred between morphologically similar classes like Sira and Dermason. |
| **Decision Tree** | Experienced a drop in accuracy (89.17%). Greedy axis-aligned split thresholds struggled slightly to carve clean boundaries between overlapping continuous feature distributions compared to continuous margins. |
| **KNN** | Performed competitively (91.66% accuracy) when utilizing standardized feature scaling, effectively grouping local geometric neighborhood clusters for most bean types. |
| **Naive Bayes** | Suffered a substantial drop in accuracy (76.39%). This highlights the limitation of the conditional independence assumption when applied to the Dry Bean dataset, where physical attributes like area, perimeter, and axis lengths are heavily correlated. |
| **Support Vector Machine** | Secured the highest overall performance across every metric (92.21% accuracy, 0.9937 AUC). Optimal hyperplane margins and kernel transformations successfully handled smooth class boundaries and overlaps. |

---

### Overall Winner for the Dataset
**Support Vector Machine (SVM)** is the overall winner, achieving top-tier results across Accuracy (92.21%), AUC Score (0.9937), and MCC (0.9059) by effectively leveraging maximum-margin hyperplanes to separate complex morphological grain geometries.