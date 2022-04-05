#%%

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, WheelZoomTool, ResetTool, PanTool
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import PuRd6, Viridis6, Bokeh6

import scipy.io
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from sklearn.metrics import roc_auc_score
from pyod.models.vae import VAE
from pyod.models.suod import SUOD


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
    p.xaxis.axis_label = "Principal Component 1"
    p.yaxis.axis_label = "Principal Component 2"
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
    cfactor.sort()

    #marker map
    markers = ['circle', 'triangle']
    mfactor = ['Inlier', 'Outlier']

    #color palettes
    npal = ('#BBB0AF','#C1A6A6','#E25656','#C62A2A','#E71D1D','#FF0000')
    opal = ('#787878','#C1A6A6','#46B1A8','#ffca3a','#500EB3','#FF0000')

    p = figure(tools = [hover, WheelZoomTool(), ResetTool(), PanTool()], 
        title='Plot of Predicted Outliers for {} on the {} Dataset'.format(name, dname))

    fmap = factor_cmap('svotes', palette=opal, factors=cfactor)
    p.scatter(
        'PC1', 'PC2', source = source, size = 5,
        marker = factor_mark('predout',markers,mfactor),
        color = fmap, 
        legend_group = 'svotes')
    p.xaxis.axis_label = "Principal Component 1"
    p.yaxis.axis_label = "Principal Component 2"
    p.legend.title = "Total Outlier Votes"
    p.add_layout(p.legend[0], 'right')
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
    
    return x_pca, a_out, data

def runModel(x, method, name, dname, a_out):
    print("For the {} dataset:".format(dname))
    #create the model
    clf = method

    #fit the data
    clf.fit(x)

    #raw outlier scores
    y_scores = clf.decision_function(x)

    #generate predictions
    y_preds = clf.fit_predict(x)

    #generate probabilities
    proba = clf.predict_proba(x)

    df = pd.DataFrame(x.copy())
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
    
    return df

def getPreds(model, X):
    clf = model
    preds = clf.fit_predict(X)
    return preds

def validate(x):
    done = False
    while not done:
        try:
            if x not in ['y', 'n', 'Y', 'N']:
                x = input("Please Select Y or N: ")
            else:
                done = True
        except ValueError:
            x = input("Entry Must be non-numeric: ")
    return x

def main():
    #set contamination level
    contam = float(input("Enter the Contamination Level of the Data (0.0-0.5): "))

    # Combination Estimators
    modlist = [
            HBOS(contamination=contam),
            OCSVM(contamination=contam), 
            IForest(contamination=contam),
            MCD(contamination=contam),  
            VAE(encoder_neurons=[32,25,10,12,2],decoder_neurons=[2,12,10,25,32], contamination=contam)
            ]

    clf = SUOD(base_estimators=modlist, contamination=contam)

    #model names
    model_names = ['HBOS', 'IForest',
                'MCD', 'OCSVM', 'VAE']

    #create transformed data set
    data = input("Enter Dataset to Detect Outliers: ")
    dname = data

    x_pca, a_out, orig_data = createData(data)

    #create list of model outlier predictions
    pred_list = list()

    #run Models for outlier votes
    for model, name in zip(modlist, model_names):
        pred = getPreds(model, x_pca)
        pred_list.append(pred)

    #create dataframe of predictions
    pred_list = np.array(pred_list)
    pred_list = pred_list.transpose()
    pred_df = pd.DataFrame(pred_list)
    pred_df.columns = model_names

    pred_df['outlier_votes'] = pred_df.sum(axis=1)

    #final model run
    modname = "Ensemble Method"
    df = runModel(x_pca, clf, modname, dname, a_out)
    df["outlier_votes"] = pred_df["outlier_votes"]

    vals = [str(x) for x in df['outlier_votes']]
    df['svotes'] = vals

    #add outlier predictions and votes to original data
    orig_data['outlier_pred'] = df['predictions']
    orig_data['total_outlier_votes'] = df['outlier_votes']

    #subset of dataframe (predicted outliers only)
    out_only = orig_data[orig_data['outlier_pred']==1]

    #subset of dataframe (predicted outliers only)
    majority_votedf = orig_data[orig_data['total_outlier_votes']>=1]

    #visualizations
    predplot(df, modname, dname)
    voteplot(df, modname, dname)

    #export dataframes
    odata_filename = dname + "_with_outliers.csv"
    justo_filename = dname + "_pred_outliers.csv"
    mv_filename = dname + "_majority_voted_outliers.csv"

    origout = validate(input("Would you Like a CSV of your data with predicted outlier labels (Y or N):\n"))
    if origout.lower() == 'y':
        orig_data.to_csv(odata_filename)

    justout = validate(input("Would you Like a CSV of only the predicted outliers from your data? (Y or N):\n"))
    if justout.lower() == 'y':
        out_only.to_csv(justo_filename)

    mvout = validate(input("Would you Like a CSV of only the outliers by majority vote from your data? (Y or N):\n"))
    if mvout.lower() == 'y':
        majority_votedf.to_csv(mv_filename)


if __name__ == "__main__":
    main()






# %%

# %%
