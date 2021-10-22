#! /usr/bin/env python3

### PARSE ARGUMENTS AND DO APPROPRIATE CHECKS ###

# command-line parsing module
import argparse

# assign to variable parser the argparse class
parser = argparse.ArgumentParser(description='TOPOS: Tissue-of-Origin Predictor of Onco-Samples. A robust SVM classifier of the cancer type of primary, metastasis and cell line samples.')

# add_argument() method is what we use to specify which command-line options the program is willing to accept.
parser.add_argument('prediction_mode',help='If "pretrained", a pretrained model for the 110 most informative genes  will be loaded. If "retrained", a new model will be trained. "retrain" mode needs to be chosen if information for any of the 110 most informative genes is missing. In order to use retrain, you\'ll have to download the training matrix "df_train_prim_met_lines.pkl"')
parser.add_argument('n_genes',help='Number of most informative genes to consider. If the prediction mode is "pretrained", n_genes will be set to 110 automatically, regardless of what the user writes. If the prediction mode is "retrained", n_genes must be between 1 and 14481.')
parser.add_argument('normalization',help='Normalization method. Can be either "self" or "train". "self" is recommended to ameliorate batch effects.')
parser.add_argument('input_matrix',help='Input tsv file with data of cancer samples to predict. Matrix rows correspond to samples and matrix columns correspond to genes, which must be identified with Entrez ids. Expression values must be TPM. See sample files provided ("X_primary.tsv" and "X_met.tsv").')
parser.add_argument('output_predictions',help='Output tsv file with cancer type predictions for samples.')
parser.add_argument('-v','--verbose',help='If True, TOPOS will print explanatory messages. It defaults to False')

# parse_args() method returns some data from the options specified
args = parser.parse_args()

if (args.prediction_mode != 'pretrained') & (args.prediction_mode != 'retrained'):
	raise ValueError('mode argument should be either "pretrained" or "retrained"; "{}" is not understood.'.format(args.prediction_mode))

if (args.normalization != 'self') & (args.normalization != 'train'):
	raise ValueError('normalization argument should be either "self" or "train"; "{}" is not understood.'.format(args.normalization))

# First, the try clause (the statement(s) between the try and except keywords) is executed
# If no exception occurs, the except clause is skipped and execution of the try statement is finished
# If an exception occurs during execution of the try clause, the rest of the clause is skipped. Then if its type matches the exception named after the except keyword, the except clause is executed, and then execution continues after the try statement.
try:
	if (args.prediction_mode == 'retrained'):
		if (int(args.n_genes) < 1) | (int(args.n_genes) > 14481):
			raise ValueError
except ValueError:
	raise ValueError('n_genes argument should be an integer between 1 and 14481; "{}" is not'.format(args.n_genes))

if args.verbose is not None: # if we have assigned a value to the verbose option
	if args.verbose == 'True':
		args.verbose = True
	elif args.verbose == 'False':
		args.verbose = False
	else:
		raise ValueError('Optional verbose argument must be set to True or False; "{}" is neither.'.format(args.verbose))


### LOAD NEEDED MODULES ###

import pandas as pd # package for handling dataframes
from sklearn.preprocessing import StandardScaler,LabelEncoder # StandardScaler: change raw feature vectors into a representation
                                                              # that is more suitable for the downstream estimators
                                                              # LabelEncoder: Encode target labels with value between 0 and
                                                              # n_classes-1
from sklearn.svm import SVC # support vector classifier

### READ USER'S DATA AND PREPROCESS IT ###

if args.verbose:
	print('\n... Running TOPOS: Tissue-of-Origin Predictor of Onco-Samples ...\n',flush=True)
	

# import user gene expression matrix
# cols = genes (named with Entrez ids)    rows = samples (named)

#                  7105    64102
# X22RV1_PROSTATE  0.00     0.02
# X2313287_STOMACH 0.00     0.00

df_user = pd.read_table(args.input_matrix, index_col=0) # Column(s) to use as the row labels of the DataFrame, either given as
                                                        # string name or column index
df_user.columns = [str(elem) for elem in df_user.columns] # convert into strings user's input matrix column names

# import table with ranked entrez ids of gene signature
# ranking    entrez_id
# 1          7849
# 2          4072

df_ranking = pd.read_table('./required_data/rfe_ranking.tsv')
df_ranking['entrez_id'] = df_ranking['entrez_id'].astype(str) # convert the values in column 'entrez_id' into strings

if args.prediction_mode == 'pretrained':
	args.n_genes = 110
    # subset ranking df taking only top 110 genes
	best_n_genes = df_ranking.loc[df_ranking['ranking'] <= int(args.n_genes),'entrez_id']
	if (~best_n_genes.isin(df_user.columns)).any(): # if columns' names of df_user are not in best_n_genes df
                                                    # ~ means bitwise "not", inversing boolean mask
		raise ValueError('The following genes are not provided. Expression values for these genes are required when choosing the prediction mode "pretrained". If you do not have this information, use the prediction mode "retrained" instead. You will also need to download the training matrix. Entrez ids: {}'.format(best_n_genes.loc[~best_n_genes.isin(df_user.columns)].values))

else: # in case we are in the 'retrain' mode
      # subset ranking df taking top "n_genes" genes
	best_n_genes = df_ranking.loc[df_ranking['ranking'] <= int(args.n_genes),'entrez_id']
    # subset in turn best_n_genes taking only genes which are also in user_df columns
	best_n_genes = best_n_genes.loc[best_n_genes.isin(df_user.columns)]

# final subset of user_df gene expression matrix
X_user = df_user.loc[:,best_n_genes]

if args.normalization == 'self':
	if args.verbose:
		print('1. Normalizing data matrix according to "self" method. The user\'s data will be scaled so that each feature has a mean of 0 and a standard deviation of 1 across samples. Such a normalization may help to reduce batch effects if the user\'s samples come from different distribution than the training distribution.', flush=True)
    # class StandardScaler --> to compute the mean and standard deviation
	scaler = StandardScaler().fit(X_user)
    # transform user_df dataframe
	X_user_scl = pd.DataFrame(scaler.transform(X_user), index=X_user.index, columns=X_user.columns)

else: # if normalization strategy is set to 'train'
	if args.verbose:
		print('1. Normalizing data matrix according to "train" method. The user\'s data will be scaled in the same way as the primary training data. This is the correct behaviour if the user\'s samples come from the same distribution as the training distribution. However, if this is not the case, the "self" normalization method is recommended instead in order to avoid batch effects.', flush=True)
    
    # import table with mean and standard deviation for each gene used in the training procedure
    #         mean               std
    # 388795 2.5848756084369904  12.799980617989789
    # 822574 0.1405877771758     411.13135097759596
	genes_mean_std = pd.read_table('./required_data/genes_mean_and_std.tsv',index_col=0)
	genes_mean_std.index = [str(elem) for elem in genes_mean_std.index] # convert into strings each entrez_id in rows
    # transform user_df dataframe using gene specific mean and standard deviation derived from the training
	X_user_scl = (X_user - genes_mean_std.loc[X_user.columns,'mean'])/(genes_mean_std.loc[X_user.columns,'std'])

    
### LOAD PRETRAINED CLASSIFIER OR TRAIN CLASSIFIER ON THE GENES AVAILABLE ###

if args.prediction_mode == 'pretrained':
	from sklearn.externals import joblib # save a model in scikit-learn by using Python’s built-in persistence model pickle
                                         # joblib is a replacement of pickle (dump & load) which is more efficient on objects
                                         # that carry large numpy arrays internally as is often the case for fitted scikit-learn
                                         # estimators, but can only pickle to the disk and not to a string
	if args.verbose:
		print('2. Loading pretrained classifier on 110 genes.',flush=True)
	clf = joblib.load('./required_data/clf_pretrained_110.pkl') # load the pickled trained model
	encoder = joblib.load('./required_data/label_encoder.pkl') # load the encoded target values, i.e. Y
else:
	if args.verbose:
		print('2. Training classifier on {} genes. This number may be smaller than the selected number ({}) if genes in the training matrix are not provided in the user\'s matrix, or if the selected number of genes is larger than 2000. The latter is because the gene ranking obtained with RFE does not sort the least important genes one by one, but rather puts them in buckets of decreasing importance. Otherwise RFE would be prohibitive to compute.'.format(X_user_scl.shape[1], args.n_genes), flush=True)
        
    # load training matrix
	df_train = pd.read_pickle('./required_data/df_train_prim_met_lines.pkl')
    # subset the training matrix
    # - everything except last column (probably labels are on the last column)
	X_train_scl = df_train.iloc[:,:-1]
    # - all rows for which genes are in best_n_genes
	X_train_scl = X_train_scl.loc[:,best_n_genes]
    
    # encode Y labels
	encoder = LabelEncoder().fit(df_train.iloc[:,-1])
    # normalize Y labels
	Y_train = encoder.transform(df_train.iloc[:,-1]) # encoder.transform([1, 1, 2, 6]) --> array([0, 0, 1, 2]...)
    # train the model
	clf = SVC(kernel='linear', C=1).fit(X_train_scl,Y_train)

### PREDICT CANCER TYPES OF USER'S SAMPLES ###

if args.verbose:
	print('3. Predicting the cancer type of the user\'s samples.',flush=True)

# create dataframe with TOPOS predictions, rownames = samples in X_user_scl
# clf.predict(X_user_scl): returns the class labels for samples in X_user_scl
# encoder.inverse_transform(): from normalized to original class labels
P_user = pd.Series(encoder.inverse_transform(clf.predict(X_user_scl)),index=X_user_scl.index)

# write P_user into a tab-separated file
P_user.to_csv(args.output_predictions,sep='\t')

if args.verbose:
	print('\n... Wrote predictions to {}. Thank you for using TOPOS ...\n'.format(args.output_predictions), flush=True)
