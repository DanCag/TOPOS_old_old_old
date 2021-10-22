#! /usr/bin/env python3


# --------------- #
# Parse arguments #
# --------------- #

# command-line parsing module
import argparse 

# Assign to variable parser the argparse class
parser = argparse.ArgumentParser(
        
    description = ('TOPOS: Tissue-of-Origin Predictor of Onco-Samples.\n' +  
                   'A robust SVM classifier of the cancer type of' + 
                   'primaries, metastases, cell lines' +
                   'and circulating tumor cells.')
    )

# add_argument() method is what we use to specify 
# which command-line options the program is willing to accept.

# verbosity (optional argument)
parser.add_argument(
    '-v',
    '--verbose',
    help = 'If True, TOPOS will print explanatory messages.' +
           '(default is False)'
           )

# number of genes (optional argument)
parser.add_argument(
    '-n', 
    '--n_genes', 
    default = None, 
    type = int,
    help = ('Number of most informative genes to consider.\n' + 
            '"n_genes" must be between 1 and 14369 (default is None')
    )

# normalization strategy
parser.add_argument(
    'normalization',
    help = (
        'Normalization method. Can be either "self" or "train":\n\n' +     
        '- In "self" the normalization is done' + 
        ' irrespectively of the training matrix\n' + 
        '- In "train", the gene-wise step is performed considering' + 
        ' the mean and standard deviation' + 
        ' of the sample-wise standardized training matrix')
    )

# user gene expression matrix
parser.add_argument(
    'input_matrix',
    help = (
        'Input tsv file with data of cancer samples to predict.\n' + 
        'Rows are samples, while columns are genes (Entrez ids).\n' + 
        'Expression values must be TPM (see sample files provided).'
        )
    )

# output path
parser.add_argument(
    'output_predictions', 
     help = 'Output tsv file with cancer type predictions for samples.'
     )


# parse_args() method returns some data from the options specified
args = parser.parse_args()


# ------------------------------------- #
# Check if options can be read by TOPOS #
# ------------------------------------- #

## n_genes ##

try: 
    if args.n_genes is not None:
        if (args.n_genes < 1) | (args.n_genes > 14369):
             raise ValueError
except ValueError:
    raise ValueError(
        '"n_genes" argument should be an integer between 1 and 14369;' + 
        '"{}" is not'.format(args.n_genes)
        )

## normalization ##

if (args.normalization != 'self') & (args.normalization != 'train'):
    raise ValueError(
        'normalization argument should be either "self" or "train";' + 
        '"{}" is not understood.'.format(args.normalization)
        )

## verbosity ##

if args.verbose is not None: # if we have assigned a value to the verbose option
    if args.verbose == 'True':
        args.verbose = True
    elif args.verbose == 'False':
        args.verbose = False
    else:
        raise ValueError(
            'Optional verbose argument must be set to True or False;' +
            '"{}" is neither.'.format(args.verbose)
            )


# ---------- #
# Parameters #
# ---------- #

# Paths
function_dir = './required_data/functions'
train_path =  (
        './required_data/training/' +
        'tcga-met500-ccle_training_14369-genes_tpm.pkl'
        )
ranking_path = './required_data/rfe_ranking_500-125-1.tsv'
encoder_path = './required_data/encoder_tissues-15.pkl'
labels_path = './required_data/training/labels_tcga-met500-ccle_training.pkl'


# -------- #
# Packages #
# -------- #

# pandas
import pandas as pd

# my functions
import sys

# add path-with-module to python path at runtime
sys.path.append(function_dir)

# function that computes mean and sd for each gene
from mean_sd import mean_sd

# sklearn
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.svm import SVC


# --------- #
# Read data #
# --------- #

if args.verbose:
    print(
        '\n... Running TOPOS: Tissue-of-Origin Predictor of Onco-Samples ...\n',
        flush=True
        )

## Training ##

# load training matrix
df_train = pd.read_pickle(train_path)

# convert columns into strings
df_train.columns = [str(elem) for elem in df_train.columns]

## Testing ##

# load user gene expression matrix
df_user = pd.read_table(args.input_matrix, 
                        index_col = 0)
# convert columns into strings
df_user.columns = [str(elem) for elem in df_user.columns]


## Ranking ##

# import table with ranked entrez ids of gene signature
df_ranking = pd.read_table(ranking_path)

# convert the values in column 'entrez_id' into strings
df_ranking['entrez_id'] = df_ranking['entrez_id'].astype(str) 


# ------------ #
# Process data #
# ------------ #

## Features ##

# features in common between train and user
overlap_genes = df_train.columns[df_train.columns.isin(df_user.columns)]

if args.n_genes is not None:

    # subset df_ranking taking top 'n_genes'
    best_n_genes = df_ranking.loc[
            df_ranking['ranking'] <= args.n_genes,
            'entrez_id'
            ]
    # subset best_n_genes to take only those genes that are in df_user columns too
    best_n_genes = best_n_genes.loc[best_n_genes.isin(df_user.columns)]

else: 

    best_n_genes = overlap_genes

# subset df_train and df_user by common genes
df_train = df_train.loc[:, overlap_genes]
df_user = df_user.loc[:, overlap_genes] 


## Normalization ##

# Training  
                    
# sample-wise standardization
X_train_sw = pd.DataFrame(scale(X = df_train, 
                                axis = 1), 
                          index = df_train.index, 
                          columns = df_train.columns)
                          
# compute mean and standard deviation of X_train_sw
mean_sd_train_df = mean_sd(X_train_sw)

# transform train_df dataframe 
# using gene specific mean and standard deviation derived from the training
X_train_sw_gw = (
        (X_train_sw - mean_sd_train_df.loc[X_train_sw.columns, 'mean'])/
        (mean_sd_train_df.loc[X_train_sw.columns, 'sd'])
        )

# call normalized testing set as 'X_train_scl'
X_train_scl = X_train_sw_gw


# Testing

if args.normalization == 'self':

    if args.verbose:
        print(
            '1. Normalizing data matrix according to "self" method.\n',
            '------------------------------------------------------\n\n',
            'Normalization on user\'s data is idependent from training matrix\n',
            flush=True
            )

    # sample-wise standardization
    X_user_sw = pd.DataFrame(scale(X = df_user, 
                                   axis = 1), 
                             index = df_user.index, 
                             columns = df_user.columns)
   
    # compute mean and standard deviation of X_user_sw
    # by calling mean_sd function
    mean_sd_user_df = mean_sd(X_user_sw)

    # gene-wise standardization
    X_user_sw_gw = (
            (X_user_sw - mean_sd_user_df.loc[X_user_sw.columns,'mean'])/
            (mean_sd_user_df.loc[X_user_sw.columns,'sd'])
            )
   
    # call normalized testing set as 'X_user_scl'
    X_user_scl = X_user_sw_gw

else: # if normalization strategy is set to 'train'

    if args.verbose:
        print(
            '1. Normalizing data matrix according to "train" method.\n',
            '-------------------------------------------------------\n\n',
            'The user\'s data is scaled sample-wise,\n',
            'while the gene-wise standardization is performed\n', 
            'using the mean and sd of the sample-wise scaled training matrix\n', 
            flush=True)

    # independent sample-wise standardization
    X_user_sw = pd.DataFrame(scale(X = df_user, 
                                   axis = 1), 
                             index = df_user.index, 
                             columns = df_user.columns)
                            
    # Gene-wise standardization

    # transform user_df dataframe using 
    # gene specific mean and standard deviation derived from the training
    X_user_sw_gw = (
            (X_user_sw - mean_sd_train_df.loc[X_user_sw.columns,'mean'])/
            (mean_sd_train_df.loc[X_user_sw.columns,'sd'])
            )
   
    # call normalized testing set as 'X_user_scl'
    X_user_scl = X_user_sw_gw
   
   

# ---------- #    
# Classifier #
# ---------- #

    
# load the encoder
encoder = pd.read_pickle(encoder_path) 

# subset X_train_scl and X_user_scl taking best_n_genes
# that can either be the overlap between training and testing features
# or the selected top k genes available in df_user 
X_train_scl = X_train_scl.loc[:, best_n_genes]
X_user_scl = X_user_scl.loc[:, best_n_genes]

if args.verbose: 
    print(
        '2. Training classifier on {} genes\n'.format(X_user_scl.shape[1]),
        '----------------------------------\n\n',

        'This number can be\n\n',
        'a) bigger if "n_genes" option is not defined\n\n', 
        '   This is because the overlap between training\n',
        '   and testing matrix will be considered.\n\n',
        'b) smaller than the selected number ({})\n'.format(args.n_genes),
        '  if genes in the training matrix are not provided in the user\'s matrix,\n', 
        '  or if the selected number of genes is larger than 494.\n\n',
        '  This is because the gene ranking obtained with RFE\n',
        '  does not sort the least important genes one by one,\n',
        '  but rather puts them in buckets of decreasing importance.\n', 
        '  Otherwise RFE would be prohibitive to compute.\n', 
        flush=True
        )
        
# load labels (pandas series)
labels_train = pd.read_pickle(labels_path)

# encode Y labels
encoder = LabelEncoder().fit(labels_train)
# transform Y labels
Y_train = encoder.transform(labels_train)    

# train the model
clf = SVC(kernel = 'linear', C = 1).fit(X_train_scl, Y_train)


# ---------- #
# Prediction # 
# ---------- #

# Predict cancer types of user's samples

if args.verbose:
    print('3. Predicting the cancer type of the user\'s samples.\n',
            flush=True)

# create dataframe with TOPOS predictions (rownames = samples in X_user_scl)
# clf.predict(X_user_scl): returns the class labels for samples in X_user_scl
# encoder.inverse_transform(): from normalized to original class labels
P_user = pd.Series(encoder.inverse_transform(clf.predict(X_user_scl)),
                   index = X_user_scl.index, 
                   name = 'prediction')

# write P_user into a tab-separated file
P_user.to_csv(args.output_predictions,
              sep='\t')

if args.verbose:
    print(
        '... Wrote predictions to {}'.format(args.output_predictions),
        '...\n\n', 
        '!!! Thank you for using TOPOS !!!\n', 
        flush=True)

