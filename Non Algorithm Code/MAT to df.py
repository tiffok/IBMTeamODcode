
import scipy.io
import pandas as pd

#read in mat file
udata = scipy.io.loadmat("lympho.mat")


# create dataframe (add array for column names separately)
data_val = udata['X']

data = pd.DataFrame(data_val)

#actual outlier labels
actuals = pd.DataFrame(udata['y'])

#total data frame (including labeled outliers)

data['Y'] = actuals

#outlier column
a_out = data['Y']

print(data.head())

#drop outlier label column
data.drop('Y', inplace=True, axis=1)


print(data.head())



