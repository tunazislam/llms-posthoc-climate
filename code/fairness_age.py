import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Example: Load your classified dataset with true and predicted labels along with demographics
data = pd.read_csv('result/target_age_ads.csv')

# Example structure of data
# data = pd.DataFrame({
#     'ad_id': [1, 2, 3, 4],
#     'true_gender': ['Male', 'Female', 'Male', 'Female'],
#     'predicted_gender': ['Male', 'Female', 'Female', 'Male'],
#     'true_age_group': ['Young', 'Senior', 'Early Working', 'Late Working'],
#     'predicted_age_group': ['Young', 'Senior', 'Young', 'Late Working']
# })


# Calculate confusion matrix for age group
conf_matrix_age = confusion_matrix(data['true_age_group'], data['predicted_age_group'])
print("Confusion Matrix for Age Group:\n", conf_matrix_age)

# Classification report for age group
report_age = classification_report(data['true_age_group'], data['predicted_age_group'])
print("Classification Report for Age Group:\n", report_age)

# Fairness Metric Calculation (Demographic Parity)
age_counts = data['true_age_group'].value_counts()
age_preds = data['predicted_age_group'].value_counts()
demographic_parity = age_preds / age_counts
print("Demographic Parity for Age Group:\n", demographic_parity)

# Define a function to calculate Equal Opportunity and Predictive Equality

# Define labels
labels = ['Early working', 'Late working', 'Senior', 'Young']

# Calculate TPR and FPR for each class
def calculate_tpr_fpr(confusion_matrix):
    n_classes = confusion_matrix.shape[0]
    tpr = []
    fpr = []
    
    for i in range(n_classes):
        # True Positives (TP) for class i
        tp = confusion_matrix[i, i]
        
        # False Negatives (FN) for class i
        fn = np.sum(confusion_matrix[i, :]) - tp
        
        # False Positives (FP) for class i
        fp = np.sum(confusion_matrix[:, i]) - tp
        
        # True Negatives (TN) for class i
        tn = np.sum(confusion_matrix) - (tp + fn + fp)
        
        # Calculate TPR and FPR
        tpr_value = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_value = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        tpr.append(tpr_value)
        fpr.append(fpr_value)
        
        # Print the result for each class
        print(f'{labels[i]}: TPR = {tpr_value:.2f}, FPR = {fpr_value:.2f}')
    
    return tpr, fpr

# Calculate TPR and FPR for each class
tpr, fpr = calculate_tpr_fpr(conf_matrix_age)

# Print overall results
print("\nEqual Opportunity (TPR) for each class:", tpr)
print("Predictive Equality (FPR) for each class:", fpr)