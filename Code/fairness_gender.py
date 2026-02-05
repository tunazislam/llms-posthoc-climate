import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Example: Load your classified dataset with true and predicted labels along with demographics
data = pd.read_csv('result/target_gender_ads.csv')

# Example structure of data
# data = pd.DataFrame({
#     'ad_id': [1, 2, 3, 4],
#     'true_gender': ['Male', 'Female', 'Male', 'Female'],
#     'predicted_gender': ['Male', 'Female', 'Female', 'Male'],
#     'true_age_group': ['Young', 'Senior', 'Early Working', 'Late Working'],
#     'predicted_age_group': ['Young', 'Senior', 'Young', 'Late Working']
# })

# Calculate confusion matrix for gender
conf_matrix_gender = confusion_matrix(data['true_gender'], data['predicted_gender'])
print("Confusion Matrix for Gender:\n", conf_matrix_gender) # [[56  3][ 7 40]]


# Classification report for gender
report_gender = classification_report(data['true_gender'], data['predicted_gender'])
print("Classification Report for Gender:\n", report_gender)

# # Calculate confusion matrix for age group
# conf_matrix_age = confusion_matrix(data['true_age_group'], data['predicted_age_group'])
# print("Confusion Matrix for Age Group:\n", conf_matrix_age)

# # Classification report for age group
# report_age = classification_report(data['true_age_group'], data['predicted_age_group'])
# print("Classification Report for Age Group:\n", report_age)

# Fairness Metric Calculation (Demographic Parity)
gender_counts = data['true_gender'].value_counts()
gender_preds = data['predicted_gender'].value_counts()
demographic_parity = gender_preds / gender_counts
print("Demographic Parity for Gender:\n", demographic_parity)

# Define a function to calculate Equal Opportunity and Predictive Equality
# Confusion matrix values (format: [[TN, FP], [FN, TP]])

# Extract values from confusion matrix
TN_female, FP_female = conf_matrix_gender[0]  # For females
FN_male, TP_male = conf_matrix_gender[1]      # For males

# Calculate True Positives and False Negatives for Female
TP_female = conf_matrix_gender[0][0]  # 56
FN_female = conf_matrix_gender[1][0]  # 7

# Calculate True Positives and False Negatives for Male
TP_male = conf_matrix_gender[1][1]  # 40
FN_male = conf_matrix_gender[1][0]  # 7

# Calculate True Negatives and False Positives for Male
FP_male = conf_matrix_gender[0][1]  # 3
TN_male = conf_matrix_gender[0][0]  # 56

# Calculate True Negatives and False Positives for Female
FP_female = conf_matrix_gender[0][1]  # 3
TN_female = conf_matrix_gender[1][1]  # 40

# Equal Opportunity (True Positive Rate, TPR)
tpr_female = TP_female / (TP_female + FN_female)
tpr_male = TP_male / (TP_male + FN_male)

# Predictive Equality (False Positive Rate, FPR)
fpr_female = FP_female / (FP_female + TN_female)
fpr_male = FP_male / (FP_male + TN_male)

# Print the results
print(f"Equal Opportunity (TPR) for Female: {tpr_female:.2f}")
print(f"Equal Opportunity (TPR) for Male: {tpr_male:.2f}")
print(f"Predictive Equality (FPR) for Female: {fpr_female:.2f}")
print(f"Predictive Equality (FPR) for Male: {fpr_male:.2f}")