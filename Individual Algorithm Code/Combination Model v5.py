#%%

from bokeh.plotting import figure, show
import panel as pn
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import Category10, Viridis6, Bokeh6

import scipy.io
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.metrics import roc_auc_score
from pyod.models.vae import VAE
from pyod.models.suod import SUOD

def actualplot(a_out):
    print("Actual Outlier Counts: 0: inliers 1: outliers")
    print(a_out.value_counts())
    print("plot of actual outliers")
    sns.scatterplot(data = x_pca, x = 'PC1', y = 'PC2', hue = a_out)
    plt.show()


def predplot(df, name, dname):
    source = ColumnDataSource(df)
    hover = HoverTool(
        tooltips = [
            ("Anomaly Score", "@scores"),
            ("Outlier Classification", "@predout"),
            ("Total Outlier Votes", "@outlier_votes"),
            ("Outlier Classification Status", "@class_status")
        ]
    )
    
    cfactor = df['predout'].unique()
    
    npal = ('#ff7f0e','#1f77b4')
    rpal = ('#1f77b4','#ff7f0e')
    p = figure(tools = [hover, WheelZoomTool(), ResetTool(), PanTool()], title='Plot of Predicted Outliers for {} on the {} Dataset'.format(name, dname))
    p.scatter(
        'PC1', 'PC2', source = source, 
        color = factor_cmap('predout', palette=npal, factors=cfactor),
        legend_group = 'predout')
    show(p)

def voteplot(df, name, dname):
    source = ColumnDataSource(df)
    #attributes of hover tool for plot
    hover = HoverTool(
        tooltips = [
            ("Anomaly Score", "@scores"),
            ("Outlier Classification", "@predout"),
            ("Total Outlier Votes", "@outlier_votes"),
            ("Outlier Classification Status", "@class_status")
        ]
    )
    
    #factors used for coloring points
    cfactor = df['svotes'].unique()

    #marker map
    markers = ['circle', 'triangle']
    mfactor = ['Inlier', 'Outlier']

    test = ['svotes', 'predout']

    p = figure(tools = [hover, WheelZoomTool(), ResetTool(), PanTool()], 
        title='Plot of Predicted Outliers for {} on the {} Dataset'.format(name, dname))

    fmap = factor_cmap('svotes', palette=Viridis6, factors=cfactor)
    p.scatter(
        'PC1', 'PC2', source = source, size = 5,
        marker = factor_mark('predout',markers,mfactor),
        color = fmap, 
        legend_group = 'svotes')
    
    p.legend.title = "Total Outlier Votes"
    p.add_layout(p.legend[0], 'right')
    #p.step(source = source, legend_group = 'predout')
    show(p)


def modelEvals(a_out, y_pred, y_score, df, name):
    
    print("For the {} method: ".format(name))
    #confusion matrix
    from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
    y_true = a_out
    
    #display confusion matrix
    c_mat = confusion_matrix(y_true, y_pred, labels=[0,1])
    c_disp = ConfusionMatrixDisplay(confusion_matrix=c_mat, display_labels=[0,1])
    c_disp.plot(cmap="bone_r")
    plt.show()


    tn = c_mat[0,0]
    fp = c_mat[0,1]
    fn = c_mat[1,0]
    tp = c_mat[1,1]

    actno = tn + fp
    actyes = tp + fn
    #rates
    tpr = tp / actyes
    fpr = fp / actno
    if fp + tp != 0:
        precision = tp / (fp + tp)
    accuracy = (tp + tn) / (actno + actyes)
    prevalence = actyes / (actno + actyes)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(a_out,y_score)

    #print results
    print("There are {} True Negatives.".format(tn))
    print("There are {} False Positives.".format(fp))
    print("There are {} False Negatives.".format(fn))
    print("There are {} True Positives.".format(tp))
    print("Accuracy = {:.2f}".format(accuracy))
    if fp + tp != 0:
        print("Precision = {:.2f}".format(precision))
    print("Prevalence = {:.2f}".format(prevalence))
    print("True Positive Rate = {:.2f}".format(tpr))
    #print("False Positive Rate = " + str(fpr))
    print("F1 Score = {:.2f}".format(f1) )
    print("AUC = {:.2f}".format(auc))
    print("\n")

    #plot ROC curve
    probab = df['probabilities']
    nfpr, ntpr, threshold = roc_curve(a_out, probab, pos_label=0)
    plt.plot(nfpr, ntpr, color = "darkorange", label = f"Model: AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve for Outlier Detection Method")
    plt.legend(loc="lower right")
    plt.show()

def createData(str):
    #read in data
    #read in mat file
    udata = scipy.io.loadmat(str)


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
    
    return x_pca, a_out

def runModel(x, method, name, dname):
    print("For the {} dataset:".format(dname))
    #create the model
    clf = method

    #fit the data
    clf.fit(x_pca)

    #raw outlier scores
    y_scores = clf.decision_function(x_pca)

    #generate predictions
    y_preds = clf.fit_predict(x_pca)

    #generate probabilities
    proba = clf.predict_proba(x_pca)

    df = pd.DataFrame(x_pca.copy())
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
    print("Predicted Outlier Counts: 0: inliers 1: outliers")
    print(y_pred.value_counts())


    #predplot(df, name, dname)
    #actualplot(a_out)
    modelEvals(a_out,y_pred, y_scores, df, name)
    
    auc = roc_auc_score(a_out, y_scores)
    return auc, df

def getPreds(model, X):
    clf = model
    preds = clf.fit_predict(X)
    return preds


#set contamination level
contam = 0.032

# Combination Estimators
base_estimators2 = [
        HBOS(contamination=contam),
        OCSVM(contamination=contam), 
        IForest(contamination=contam),
        MCD(contamination=contam),  
        VAE(encoder_neurons=[32,25,10,12,2],decoder_neurons=[2,12,10,25,32], contamination=contam)
        ]

clf = SUOD(base_estimators=base_estimators2, contamination=contam)
#list of models
modlist = [
            HBOS(contamination=contam), 
            IForest(contamination=contam),
            MCD(contamination=contam), 
            OCSVM(contamination=contam),
            VAE(encoder_neurons=[32,25,10,12,2],decoder_neurons=[2,12,10,25,32], contamination=contam)
            ]


#model names
model_names = ['HBOS', 'IForest',
            'MCD', 'OCSVM', 'VAE']

#create transformed data set
data = "musk.mat"
dname = "Musk"

x_pca, a_out = createData(data)


pred_list = list()

#run Models for outlier votes
for model, name in zip(modlist, model_names):
    pred = getPreds(model, x_pca)
    pred_list.append(pred)

pred_list = np.array(pred_list)
pred_list = pred_list.transpose()
pred_df = pd.DataFrame(pred_list)
pred_df.columns = model_names

pred_df
pred_df['outlier_votes'] = pred_df.sum(axis=1)

#final model run
modname = "Ensemble Method"
auc, df = runModel(x_pca, clf, modname, dname)
df["outlier_votes"] = pred_df["outlier_votes"]

vals = [str(x) for x in df['outlier_votes']]
df['svotes'] = vals

predplot(df, modname, dname)
voteplot(df, modname, dname)






# %%
