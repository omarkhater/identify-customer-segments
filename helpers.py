###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.cluster import KMeans
from sklearn import preprocessing

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def compare_dist(col,data1,data2):
    """
    compare the distribution of values between two different subsets for given column
    INPUTS: 
    col: string
        Name of the feature to plot
    data1: pandas dataframe
    data2: pandas dataframe
    """
    fig, axs = plt.subplots(1,2, figsize = (15,5))
    sns.countplot(ax = axs[0], data = data1, x = col )
    sns.countplot(ax = axs[1], data = data2, x = col )

def do_pca(n_components, data):
    '''
    Transforms data using PCA to create n_components, and provides back the results of the
    transformation.

    INPUT: n_components - int - the number of principal components to create
           data - the data you would like to transform

    OUTPUT: pca - the pca object created after fitting the data
            X_pca - the transformed X matrix with new number of components
    '''
    X = StandardScaler().fit_transform(data)
    pca = PCA(n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca

def scree_plot(pca):
    '''
    Creates a scree plot associated with the principal components 
    
    INPUT: pca - the result of instantian of PCA in scikit learn
            
    OUTPUT:
            None
    '''
    num_components=len(pca.explained_variance_ratio_)
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
 
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    cumvals = np.cumsum(vals)
    ax.bar(ind, vals)
    ax.plot(ind, cumvals)
    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)
 
    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)
 
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    plt.title('Explained Variance Per Principal Component')
    return vals,cumvals         

def pca_results(full_dataset, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = full_dataset.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	#fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	#components.plot(ax = ax, kind = 'bar');
	#ax.set_ylabel("Feature Weights")
	#ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	#for i, ev in enumerate(pca.explained_variance_ratio_):
		#ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)

def map_weights(FinalPCAresults, i):
    """
    return sorted list of feature weights, for the i-th principal component given the 
    inputs: 
    - FinalPCAresults: final PCA results returned by the function pca_results ( data frame of Feature names Vs principal components)
    - i : i-th principle component. 
    Output:
    - pandas series of containing sorted weigth along with feature names.
    """
    indices = np.argsort(FinalPCAresults.iloc[i,:])
    return FinalPCAresults.iloc[i,:][np.flip(indices)]

def fit_kmeans(data, centers):
    '''
    INPUT:
        data = the dataset you would like to fit kmeans to (dataframe)
        centers = the number of centroids (int)
    OUTPUT:
        labels - the labels for each datapoint to which group it belongs (nparray)
    
    '''
    kmeans = KMeans(centers)
    Kmeans_obj = kmeans.fit(data)
    labels = kmeans.predict(data)
    return labels,Kmeans_obj

def impute_nan_most_frequent_category(DataFrame,ColName):
    """
    Impute categorical data with mode values
    """
    most_frequent_category=DataFrame[ColName].mode()[0]
    # replace nan values with most occured category
    DataFrame[ColName + "_Imputed"] = DataFrame[ColName]
    DataFrame[ColName + "_Imputed"].fillna(most_frequent_category,inplace=True)
    
def clean_data(df, feat_info):
    """
    Perform feature trimming, re-encoding, and engineering for demographics data
    
    INPUT: 
        - df: Demographics DataFrame
        - feat_info: Information about df columns
    OUTPUT: Trimmed and cleaned demographics DataFrame
    """

    print('Data cleansing pipeline...')
    print('df shape is {}'.format(df.shape))
    ### Main cleaning steps:    
    ## 1- convert missing value codes into NaNs, ...
    print('\t1-Replacing missing values per column with Nan...')
    a = list(feat_info['missing_or_unknown'])
    ToNan = [i.strip('][,').split(',') for i in a]
    for i,j in zip(df.columns, ToNan):
        NanIndices = df.query('{} == {}'.format(i, j)).index
        df.at[NanIndices,i] = np.NaN

    print('\t\tReplacing done')
    # ------------------------------------------------------------------------------------------------ 
    ## 2- Remove selected columns and rows 
    print('\t2-Removing the outliars columns and rows...')
    # 2.1- Perform an assessment of how much missing data there is in each column of the dataset.
    
    MissingValues = pd.DataFrame([df.isna().sum(),(df.isna().sum() / df.shape[0]) * 100], 
                                    index = ['Values','Per'], 
                                 columns = df.columns).T
    Threshold = MissingValues.describe()['Per']['75%']
    Todelete = MissingValues.query('Per > {}'.format(Threshold))
    DeletedColumns = df.loc[:,Todelete.index]
    print('Deleted columns shape is {}'.format(DeletedColumns.shape))
    # 2.2- Remove the outlier columns from the dataset
    
    df.drop(Todelete.index, axis = 1 , inplace = True)
    feat_info.drop(Todelete.index, axis = 0 , inplace = True)
    # 2.3- Perform an assessment of how much missing data there is in each row of the dataset

    MissingValuesRows = pd.DataFrame([df.isna().sum(axis = 1),(df.isna().sum(axis = 1) / df.shape[1]) * 100],
                                     columns = df.index, index = ['Value','Per']).T
    
    # 2.4- Divide the data into two subsets based on the number of missing values in each row.
    ThresholdRows = MissingValuesRows.describe()['Per']['75%']
    TodeleteRows = MissingValuesRows.query('Per > {}'.format(ThresholdRows))
    DeletedRows = df.loc[TodeleteRows.index,:]
    print('Deleted rows shape is {}'.format(DeletedRows.shape))
    df.drop(TodeleteRows.index, axis = 0 , inplace = True)
    df.reset_index(inplace = True , drop = True)
    print('\t\tRemoving the outliars columns and rows done')
    print('df shape is {}'.format(df.shape))
    # ------------------------------------------------------------------------------------------------
    ## 3- Select, Re-encode, and Engineer column values.
    print('\t3-Feature Engineering...')
    # 3.1- Assess categorical variables: which are binary, which are multi-level, and which one needs to be re-encoded?
    
    cat_var = df.loc[:,list(feat_info.query('type =="categorical"').index)]
    unique_per_cat = [(i,cat_var[i].nunique()) for i in cat_var.columns]
    bin_cat = [i for i in cat_var.columns if cat_var[i].nunique() == 2 and pd.api.types.is_string_dtype(cat_var[i])]
    Multi_cat = [i for i in cat_var.columns if cat_var[i].nunique() > 2]
    To_OH = pd.concat([df.loc[:, Multi_cat], df.loc[:, bin_cat]], axis = 1)
    # 3.2- Impute missing values for data frame to be onehot encoded
    for k,v in To_OH.isna().sum().items():
        if v > 0:
            impute_nan_most_frequent_category(To_OH,k)
            To_OH.drop(k, axis = 1, inplace = True)
    print('3.2: df shape is {}'.format(df.shape))    
    # 3.3- Re-encode categorical variable(s) to be kept in the analysis.
    print('To_OH columns are {}'.format(To_OH.columns))
    OH_bin_obj = preprocessing.OneHotEncoder().fit(To_OH)
    OH_cat = OH_bin_obj.transform(To_OH)
    OH_cat = pd.DataFrame(OH_cat.toarray(), columns = OH_bin_obj.get_feature_names())
    print('3.3: shape is {}'.format(df.shape))
    # 3.4- Drop categorical features
    df.drop(cat_var , axis = 1 , inplace = True)
    feat_info.drop(cat_var, axis = 0, inplace = True)
    print('3.4: df shape is {}'.format(df.shape))
    # 3.5- Append one hot encoded data 
    df = pd.concat([df, OH_cat], axis = 1)
    print('3.5: df shape is {}'.format(df.shape))
    print('3.5: OH_cat shape is {}'.format(OH_cat.shape))
    # 3.6- Return the cleaned dataframe.
    # Investigate "PRAEGENDE_JUGENDJAHRE" and engineer new variables.
    df['movement'] = df.PRAEGENDE_JUGENDJAHRE % 2 
    Wealth = [int(i[0]) if type(i) == str else i for i in df.CAMEO_INTL_2015]
    LifeStage = [int(i[1])  if type(i) == str else i for i in df.CAMEO_INTL_2015]
    CAMEO_INTL_2015 = pd.DataFrame(list(zip(Wealth, LifeStage)), columns = ['Waelth', 'LifeStage'])
    print('3.6: df shape is {}'.format(df.shape))
    # 3.7- Drop mixed features
    df.drop(feat_info.query('type =="mixed"').index, axis = 1, inplace = True)
    # 3.8- Append new features
    df = pd.concat([df,CAMEO_INTL_2015] , axis = 1)
    print('3.7: df shape is {}'.format(df.shape))
    print('\t\tFeature Engineering done')
    print('df shape is {}'.format(df.shape))
    # ------------------------------------------------------------------------------------------------
    ## 4- Remove all the missing values
    print('\t4-Removing missing values...')
    df.dropna(inplace = True)
    df.reset_index(inplace = True ,drop = True)
    print('\t\tRemoving missing values done.')
    print('df shape is {}'.format(df.shape))
    # ------------------------------------------------------------------------------------------------
    print('Cleaning pipeline done')
    return df