
#confusion matrix with rates
 from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
y_true = a_out #actual outliers
#need to have a column of the predicted outliers for the confusion matrix (saved as y_pred)
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
#misclass = (fp + fn) / (actno + actyes)
tpr = tp / actyes
fpr = fp / actno
#tnr = tn/ actno
precision = tp / (fp + tp)
accuracy = (tp + tn) / (actno + actyes)
prevalence = actyes / (actno + actyes)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(a_out,y_pred)

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

#plot ROC curve
nfpr, ntpr, threshold = roc_curve(a_out, y_pred)
plt.plot(nfpr, ntpr, color = "darkorange", label = "Model")
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()

