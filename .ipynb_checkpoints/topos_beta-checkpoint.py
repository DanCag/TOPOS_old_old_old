#! /usr/bin/env python3


## Parse arguments
# ----------------

import argparse # command-line parsing module

# Assign to variable parser the argparse class
parser = argparse.ArgumentParser(description = 'TOPOS: Tissue-of-Origin Predictor of Onco-Samples. A robust SVM classifier of the cancer type of primaries, metastasese, cell lines and circulating tumor cells.')

# add_argument() method is what we use to specify which command-line options the program is willing to accept.

# prediction mode
parser.add_argument('prediction_mode',
                    help = 'If "pretrained", a pretrained model for the 127 most informative genes  will be loaded. If "retrained", a new model will be trained. "retrain" mode needs to be chosen if information for any of the 127 most informative genes is missing. In order to use retrain, you\'ll have to download the training matrix "train_prim-met-lines_tpm.pkl"')

# number of genes
parser.add_argument('n_genes',
                    help = 'Number of most informative genes to consider. If the prediction mode is "pretrained", n_genes will be set to 127 automatically, regardless of what the user writes. If the prediction mode is "retrained", n_genes must be between 1 and 14369.')

# normalization strategy
parser.add_argument('normalization',
                    help = 'Normalization method. Can be either "self" or "train". In "self" the normalization is done irrespectively of the training matrix. In "train", the gene-wise step is performed considering the mean and standard deviation of the sample-wise standardized training matrix.')

# user gene expression matrix
parser.add_argument('input_matrix',
                    help = 'Input tsv file with data of cancer samples to predict. Matrix rows correspond to samples and matrix columns correspond to genes, which must be identified with Entrez ids. Expression values must be TPM. See sample files provided ("X_primary.tsv" and "X_met.tsv").')

# output path
parser.add_argument('output_predictions', 
                    help = 'Output tsv file with cancer type predictions for samples.')

# verbosity
parser.add_argument('-v',
                    '--verbose',
                    help='If True, TOPOS will print explanatory messages. It defaults to False')

# parse_args() method returns some data from the options specified
args = parser.parse_args()


## Check if options can be read by TOPOS
# --------------------------------------


if (args.prediction_mode != 'pretrained') & (args.prediction_mode != 'retrained'):
	raise ValueError('mode argument should be either "pretrained" or "retrained"; "{}" is not understood.'.format(args.prediction_mode))

if (args.normalization != 'self') & (args.normalization != 'train'):
	raise ValueError('normalization argument should be either "self" or "train"; "{}" is not understood.'.format(args.normalization))

# First, the try clause (the statement(s) between the try and except keywords) is executed
# If no exception occurs, the except clause is skipped and execution of the try statement is finished
# If an exception occurs during execution of the try clause, the rest of the clause is skipped. Then if its type matches the exception named after the except keyword, the except clause is executed, and then execution continues after the try statement.
try:
	if (args.prediction_mode == 'retrained'):
		if (int(args.n_genes) < 1) | (int(args.n_genes) > 14369):
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


## Packages 
# ---------

## pandas
import pandas as pd # package for handling dataframes

## my functions
import sys
# add path-with-module to python path at runtime
sys.path.append('./required_data/functions')
from mean_sd import mean_sd # function that computes mean and sd for each gene

## sklearn
from sklearn.preprocessing import scale # scale:
from sklearn.svm import SVC # support vector classifier


## Read user's data
# -----------------

if args.verbose:
	print('\n... Running TOPOS: Tissue-of-Origin Predictor of Onco-Samples ...\n',flush=True)

# import user gene expression matrix
df_user = pd.read_table(args.input_matrix, 
                        index_col = 0)
# convert columns into strings
df_user.columns = [str(elem) for elem in df_user.columns]

# import table with ranked entrez ids of gene signature
df_ranking = pd.read_table('./required_data/rfe_ranking.tsv')
# convert the values in column 'entrez_id' into strings
df_ranking['entrez_id'] = df_ranking['entrez_id'].astype(str) 


## Process user's data 
# -------------------

# Prediction mode

if args.prediction_mode == 'pretrained':
    
    # set n_genes to default best 127 genes 
    args.n_genes = 127
    # subset df_ranking taking only top 127 genes
	best_n_genes = df_ranking.loc[df_ranking['ranking'] <= int(args.n_genes),'entrez_id']
    
    # check if best_n_genes are in columns' names of df_user 
	if (~best_n_genes.isin(df_user.columns)).any():  # ~ means bitwise "not"
		raise ValueError('The following genes are not provided. Expression values for these genes are required when choosing the prediction mode "pretrained". If you do not have this information, use the prediction mode "retrained" instead. You will also need to download the training matrix. Entrez ids: {}'.format(best_n_genes.loc[~best_n_genes.isin(df_user.columns)].values))

else: # in case we are in the 'retrain' mode
    
    # subset df_ranking taking top "n_genes" genes
	best_n_genes = df_ranking.loc[df_ranking['ranking'] <= int(args.n_genes),'entrez_id']
    # subset best_n_genes to take only those genes that are in df_user columns too
	best_n_genes = best_n_genes.loc[best_n_genes.isin(df_user.columns)]

# Once you set the genes you want,
# subset df_user gene expression matrix
X_user = df_user.loc[:,best_n_genes]


# Normalization
if args.normalization == 'self':
	if args.verbose:
		print('1. Normalizing data matrix according to "self" method. The user\'s data will be first scaled sample-wise to remove any artificial variation due to biases in library preparation, sequencing method, batch effect, etc. Afterwards, the sample-wise scaled matrix will be normalized on the features\'s level so that each feature has a mean of 0 and a standard deviation of 1 across samples. This second normalization maintains numerical stability, avoids giving more weight to a feature with larger variation compared to other features and improves the speed of convergence of the optimization algorithm', flush=True)
        
    # sample-wise standardization
    X_user_sw = pd.DataFrame(scale(X = X_user, axis = 1), index = X_user.index, columns = X_user.columns)
    
    # compute mean and standard deviation of X_user_sw
    mean_sd_df = mean_sd(X_user_sw)

    # gene-wise standardization
    X_user_sw_gw = (X_user_sw - mean_sd_df.loc[X_user_sw.columns,'mean'])/(mean_sd_df.loc[X_user_sw.columns,'sd'])
    
    # call normalized testing set as 'X_user_scl' to be consistent with script
    X_user_scl = X_user_sw_gw

else: # if normalization strategy is set to 'train'
	if args.verbose:
		print('1. Normalizing data matrix according to "train" method. The user\'s data will be sample-wise scaled. The gene-wise standardization will be performed using the mean and sd of the sample-wise scaled training matrix', flush=True)
    
    # independent sample-wise standardization
    X_user_sw = pd.DataFrame(scale(X = X_user, 
                                   axis = 1), 
                             index = X_user.index, 
                             columns = X_user.columns)
    
    if args.prediction_mode == 'pretrained':
        
        # import table with mean and standard deviation for each gene used in the training procedure
        mean_sd_df = pd.read_table('./required_data/genes_mean-sd_127-genes_sw.tsv',
                                   index_col = 0)
        # convert into strings each entrez_id
        mean_sd_df.index = [str(elem) for elem in mean_sd_df.index] 
    
    else: # 'retrained' mode
        
        ## Training standardization
        
        # sample-wise standardization
        X_train_sw = pd.DataFrame(scale(X = X_train,
                                        axis = 1), 
                                  index = X_train.index, 
                                  columns = X_train.columns)
        
        # compute mean and standard deviation of X_train_sw
        mean_sd_df = mean_sd(X_train_sw)
        
        # gene-wise standardization
        X_train_sw_gw = (X_train_sw - mean_sd_df.loc[X_train_sw.columns,'mean'])/(mean_sd_df.loc[X_train_sw.columns,'sd'])
        
        # call normalized testing set as 'X_user_scl' to be consistent with script
        X_train_scl = X_train_sw_gw
        
    # Gene-wise standardization

    # transform user_df dataframe using gene specific mean and standard deviation derived from the training
    X_user_sw_gw = (X_user - mean_sd_df.loc[X_user.columns,'mean'])/(mean_sd_df.loc[X_user.columns,'sd'])
    
    # call normalized testing set as 'X_user_scl' to be consistent with script
    X_user_scl = X_user_sw_gw

    
## Classifier
# -----------

    
# load the encoder
encoder = joblib.load('./required_data/encoder_tissues.pkl') 

# check prediction mode
if args.prediction_mode == 'pretrained':
	from sklearn.externals import joblib # save a model in scikit-learn by using Python’s built-in persistence model pickle
                                         # joblib is a replacement of pickle (dump & load) which is more efficient on objects
                                         # that carry large numpy arrays internally as is often the case for fitted scikit-learn
                                         # estimators, but can only pickle to the disk and not to a string
	if args.verbose:
		print('2. Loading pretrained classifier on 127 genes.',flush=True)
        
    # load pretrained classifier     
	clf = joblib.load('./required_data/clf_127-genes_sw-gw.pkl')

else:
    
	if args.verbose:
		print('2. Training classifier on {} genes. This number may be smaller than the selected number ({}) if genes in the training matrix are not provided in the user\'s matrix, or if the selected number of genes is larger than 469. The latter is because the gene ranking obtained with RFE does not sort the least important genes one by one, but rather puts them in buckets of decreasing importance. Otherwise RFE would be prohibitive to compute.'.format(X_user_scl.shape[1], args.n_genes), flush=True)
        
    # load training matrix
	df_train = pd.read_pickle('./required_data/training/train_prim-met-lines_tpm.pkl')
    
    # load labels (pandas series)
    labels_train = pd.read_pickle('./required_data/training/labels_train_prim-met-lines.pkl')

    # subset the training matrix
    # - all rows for which genes are in best_n_genes
	X_train = df_train.loc[:, best_n_genes]
    
    # encode Y labels
	#encoder = LabelEncoder().fit(labels_train)
    # transform Y labels
	Y_train = encoder.transform(labels_train) # encoder.transform([1, 1, 2, 6]) --> array([0, 0, 1, 2]...)
    
    # train the model
	clf = SVC(kernel='linear', C=1).fit(X_train_scl,Y_train)

    
## Predict cancer types of user's samples
# ---------------------------------------

if args.verbose:
	print('3. Predicting the cancer type of the user\'s samples.',flush=True)

# create dataframe with TOPOS predictions, rownames = samples in X_user_scl
# clf.predict(X_user_scl): returns the class labels for samples in X_user_scl
# encoder.inverse_transform(): from normalized to original class labels
P_user = pd.Series(encoder.inverse_transform(clf.predict(X_user_scl)),
                   index = X_user_scl.index)

# write P_user into a tab-separated file
P_user.to_csv(args.output_predictions,
              sep='\t')

if args.verbose:
	print('\n... Wrote predictions to {}. Thank you for using TOPOS ...\n'.format(args.output_predictions), flush=True)
