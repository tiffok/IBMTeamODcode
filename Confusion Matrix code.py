
#confusion matrix with rates
from sklearn.metrics import confusion_matrix
y_true = a_out #actual outliers
#need to have a column of the predicted outliers for the confusion matrix (saved as y_pred)
c_mat = confusion_matrix(y_true, y_pred)
tn = c_mat[0,0]
fp = c_mat[0,1]
fn = c_mat[1,0]
tp = c_mat[1,1]

actno = tn + fp
actyes = tp + fn
#rates
misclass = (fp + fn) / (actno + actyes)
tpr = tp / actyes
fpr = fp / actno
tnr = tn/ actno
precision = tp / (fp + tp)
accuracy = (tp + tn) / (actno + actyes)
prevalence = actyes / (actno + actyes)

#print results
print("There are " + str(tn) + " True Negatives.")
print("There are " + str(fp) + " False Positives.")
print("There are " + str(fn) + " False Negatives.")
print("There are " + str(tp) + " True Positives.")
print("Accuracy = " + str(accuracy))
print("Misclassification rate = " + str(misclass))
print("Precision = " + str(precision))
print("Prevalence = " + str(prevalence))
print("True Positive Rate = " + str(tpr))
print("False Positive Rate = " + str(fpr))
print("True Negative Rate = " + str(tnr))
print("F1 Score = " + str(f1))