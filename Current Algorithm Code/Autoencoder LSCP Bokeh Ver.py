#%%
from bokeh.plotting import figure, show
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.transform import factor_cmap
from bokeh.palettes import Category10

import scipy.io
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.hbos import HBOS
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.lscp import LSCP


def actualplot(df):
    
    source = ColumnDataSource(df)
    hover = HoverTool(
        tooltips = [
            ("Anomaly Score", "@scores"),
            ("Outlier Classification", "@trueout")
        ]
    )
    
    cfactor = df['trueout'].unique()
    
    npal = ('#ff7f0e','#1f77b4')
    rpal = ('#1f77b4','#ff7f0e')
    p = figure(tools = [hover, WheelZoomTool(), ResetTool(), PanTool()], title='Plot of Actual Outliers')
    p.scatter(
        'PC1', 'PC2', source = source, 
        color = factor_cmap('trueout', palette=npal, factors=cfactor),
        legend_group = 'trueout')
    show(p)



def predplot(df):
    source = ColumnDataSource(df)
    hover = HoverTool(
        tooltips = [
            ("Anomaly Score", "@scores"),
            ("Outlier Classification", "@predout"),
            ("Outlier Classification Status", "@class_status")
        ]
    )
    
    cfactor = df['predout'].unique()
    
    npal = ('#ff7f0e','#1f77b4')
    rpal = ('#1f77b4','#ff7f0e')
    p = figure(tools = [hover, WheelZoomTool(), ResetTool(), PanTool()], title='Plot of Predicted Outliers')
    p.scatter(
        'PC1', 'PC2', source = source, 
        color = factor_cmap('predout', palette=npal, factors=cfactor),
        legend_group = 'predout')
    show(p)


def modelEvals(df):
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, auc


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



#read in data
#read in mat file
udata = scipy.io.loadmat("wine.mat")


# create dataframe (add array for column names separately)
data_val = udata['X']

data = pd.DataFrame(data_val)

#actual outlier labels
actuals = pd.DataFrame(udata['y'])

#total data frame (including labeled outliers)
data['Y'] = actuals

#outlier column
a_out = data['Y']


#drop outlier label column
data.drop('Y', inplace=True, axis=1)

#scale data
scaler = StandardScaler()
scaled = scaler.fit_transform(data)
scaled = pd.DataFrame(scaled)

#run PCA
pca = PCA(2)
x_pca = pca.fit_transform(scaled)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']

#run SUOD
print("LSCP method")
#lets get a bunch of models
base_estimators = [
    AutoEncoder(hidden_neurons=[25,15,10,2,10,15,25]),
    AutoEncoder(hidden_neurons=[25,10,2,10,25]),
    AutoEncoder(hidden_neurons=[25,2,2,25])]

clf = LSCP(detector_list=base_estimators)




#fit onto data
clf.fit(x_pca)

#generate anomaly scores
y_scores = clf.decision_function(x_pca)

#generate predictions
y_preds = clf.fit_predict(x_pca)

#generate probabilities
proba = clf.predict_proba(x_pca)

#dataframe of data and predictions
df = pd.DataFrame(x_pca)
df['scores'] = y_scores
df['predictions'] =y_preds
df['truth'] = a_out
df['predout'] = np.where(df['predictions']==1,"Outlier","Inlier")
df['trueout'] = np.where(df['truth']==1,"Outlier","Inlier")
df['probabilities'] = proba[:,0]

#create column for correct/incorrectly classified outliers

#sum predictions and actual classifications
fn = lambda row: row.predictions + row.truth
df['c'] = df.apply(fn, axis = 1)

#possible conditions
conditions = [
(df['c'] == 2),
(df['c'] == 1) & (df['predictions'] == 1),
(df['c'] == 1) & (df['predictions'] == 0),
(df['c'] ==0)]

class_values = [
    'Correctly Classified Outlier', 
    'Incorrectly Classified Outlier (Actual Inlier)', 
    'Incorrectly Classified Inlier (Actual Outlier)', 
    'Correctly Classified Inlier']

df['class_status'] = np.select(conditions, class_values)


y_pred = pd.Series(y_preds)
#print("Predicted Outlier Counts: 0: inliers 1: outliers")
#print(y_pred.value_counts())

predplot(df)
actualplot(df)
modelEvals(df)

# %%
