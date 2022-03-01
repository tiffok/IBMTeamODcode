
#confusion matrix with rates
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
#plot ROC curve
nfpr, ntpr, threshold = roc_curve(df['truth'], df['probabilities'], pos_label=0)
roc_auc = roc_auc_score(df['truth'], df['scores'])

plt.figure(dpi=150)
plt.plot(nfpr, ntpr, lw=1, color='darkorange', label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.title('ROC Curve for Outlier Detection Method')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.legend()
plt.show()

#predicted and actual outlier counts
print("Actual Outlier Counts: 0: inliers 1: outliers")
print(df['truth'].value_counts())

print("Predicted Outlier Counts: 0: inliers 1: outliers")
print(df['predictions'].value_counts())

#confusion matrix

y_true = df['truth']
y_pred = df['predictions']
c_mat = confusion_matrix(y_true, y_pred)
tn = c_mat[0,0]
fp = c_mat[0,1]
fn = c_mat[1,0]
tp = c_mat[1,1]

# print in matrix form
print("confusion matrix: ")
print(c_mat)
print("\n")

actno = tn + fp
actyes = tp + fn
#rates
tpr = tp / actyes
fpr = fp / actno
precision = tp / (fp + tp)
accuracy = (tp + tn) / (actno + actyes)
prevalence = actyes / (actno + actyes)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(df['truth'],df['scores'])

#print results
print("There are {} True Negatives.".format(tn))
print("There are {} False Positives.".format(fp))
print("There are {} False Negatives.".format(fn))
print("There are {} True Positives.".format(tp))
print("Accuracy = {:.2f}".format(accuracy))
print("Precision = {:.2f}".format(precision))
print("Prevalence = {:.2f}".format(prevalence))
print("True Positive Rate = {:.2f}".format(tpr))
#print("False Positive Rate = " + str(fpr))
print("F1 Score = {:.2f}".format(f1) )
print("AUC = {:.2f}".format(auc))
print("\n")
