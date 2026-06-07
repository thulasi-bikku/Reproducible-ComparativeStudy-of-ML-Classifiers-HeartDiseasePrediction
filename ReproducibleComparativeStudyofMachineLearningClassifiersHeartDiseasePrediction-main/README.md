#Cardiovascular Disease Prediction using Machine Learning
Overview

Cardiovascular disease (CVD) is the leading cause of mortality worldwide, accounting for nearly 18 million deaths annually. Early detection is critical, but traditional diagnostic methods such as echocardiograms and angiograms are resource-intensive and require specialized expertise.

This project provides a comprehensive and reproducible comparative analysis of multiple supervised machine learning models for early heart disease prediction.

Dataset
Total samples: 1,025 patients
Features: 14 clinically relevant attributes
After preprocessing: 968 samples (after outlier removal using 1.5 IQR rule)
Methodology
Data preprocessing and outlier removal (1.5 × IQR)
10-fold stratified cross-validation
Evaluation using 5 performance metrics (mean ± standard deviation)
Statistical validation using Wilcoxon signed-rank test (α = 0.05)
Models Used
Logistic Regression (LR)
Naive Bayes (NB)
Support Vector Machine (SVM - RBF Kernel)
k-Nearest Neighbors (KNN)
Decision Tree (DT)
Random Forest (RF)
Gradient Boosting (GB)
Feed-Forward Neural Network (NN)
Results
Best Model: Random Forest
Accuracy & ROC-AUC: 0.997 ± 0.009
Runner-up: Gradient Boosting
Accuracy: 0.982 ± 0.015
Statistical analysis confirms Random Forest significantly outperforms other models.
Feature Importance

Key features influencing predictions:

ca – Number of major vessels
thalach – Maximum heart rate
oldpeak – ST depression
thal – Thalassemia type
cp – Chest pain type

These features align with known cardiovascular medical insights.

Key Contributions
Reproducible ML pipeline for CVD prediction
Comparative analysis of 8 classification models
Statistical validation of model performance
Identification of clinically significant features
Ethical Considerations

This project highlights the importance of:

Responsible use of ML in healthcare
Avoiding bias in predictions
Supporting—not replacing—medical professionals
Conclusion

This work establishes a strong benchmark for machine learning-based cardiovascular disease prediction and provides insights for real-world implementation in healthcare systems.
