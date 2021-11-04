#! /usr/bin/env python3


import time
start_time = time.time()


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
# which command-line options the program will accept.

# verbosity (optional argument)
parser.add_argument(
    '-v',
    '--verbose',
    help = 'If True, TOPOS will print explanatory messages.'
    )

# number of genes (optional argument)
parser.add_argument(
    '-n', 
    '--n_genes', 
    default = None, 
    type = int,
    help = ('Number of most informative genes to consider.\n' + 
            '"n_genes" must be between 1 and 14369')
    )
    
# save trained model (optional argument)
parser.add_argument(
    '-s', 
    '--save_model',
    nargs = 3,
    default = None, 
    help = ('If not None, TOPOS will save\n\n' +
            '- array of needed genes\n' + 
            '- table with mean and standard deviation\n' +
            '  Mean and sd are computed on training matrix' + 
            '  of the pre-trained model after sample-wise normalization' + 
            '  The number of features corresponds to the overlap' + 
            '  between training and testing used' + 
            '  when computing pre-trained model\n' +
            '- pre-trained model\n\n' +
            'The three input are pathways')
    )
    
# load pre-trained model (optional argument)
parser.add_argument(
    '-l', 
    '--load_model',
    nargs = 3,
    default = None, 
    help = ('If not None, TOPOS will load:\n\n' + 
            '- array of needed genes\n' + 
            '- table with mean and standard deviation\n' +
            '  Mean and sd are computed on training matrix' + 
            '  of the pre-trained model after sample-wise normalization' + 
            '  The number of features corresponds to the overlap' + 
            '  between training and testing used' + 
            '  when computing pre-trained model\n' +
            '- pre-trained model\n\n' +
            'The three input are pathways')
    )
      

# normalization strategy
parser.add_argument(
    'normalization',
    help = (
        'normalization method: either "self" or "train":\n\n' +     
        '- in "self" the normalization is done' + 
        '  irrespectively of the training matrix\n' + 
        '- In "train", the gene-wise step is performed considering' + 
        '  the mean and standard deviation' + 
        '  of the sample-wise standardized training matrix')
    )

# user gene expression matrix
parser.add_argument(
    'input_matrix',
    help = (
        'Input tsv file with data of cancer samples to predict.\n' + 
        'Rows are samples, columns are genes (Entrez ids).\n' + 
        'Expression values must be TPM.'
        )
    )

# output path
parser.add_argument(
    'output_predictions', 
     help = 'Output tsv file with cancer type predictions for samples.'
     )


# parse_args() method returns some data from the options specified
args = parser.parse_args()


# --------------------------------------- #
# Check if arguments can be read by TOPOS #
# --------------------------------------- #

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

# os
import os

# pandas
import pandas as pd

# numpy
import numpy as np

# my functions
import sys

# add path-with-module to python path at runtime
sys.path.append(function_dir)

# function that computes mean and sd for each gene
from mean_sd import mean_sd

# sklearn
from sklearn.preprocessing import scale
from sklearn.svm import SVC

# joblib
import joblib


# --------- #
# Read data #
# --------- #

if args.verbose:
    print(
        ('\n\n\n... Running TOPOS: ' + 
         'Tissue-of-Origin Predictor of Onco-Samples ...\n\n\n'),
        flush=True
        )

## Training ##

if args.load_model is None:

    # if user does not load a pre-trained model,
    # he needs the training matrix

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

if args.load_model is None:

    # if we load a pre-trained model, 
    # the number of genes depends on it

    if args.n_genes is not None:

        # we need ranking info 
        # only if we specify the number of genes 
        # we are interested in

        # import table with ranked entrez ids of gene signature
        df_ranking = pd.read_table(ranking_path)

        # convert the values in column 'entrez_id' into strings
        df_ranking['entrez_id'] = df_ranking['entrez_id'].astype(str) 


# ------------ #
# Process data #
# ------------ #

if args.verbose:
    print(
        ('1. Processing data\n' + 
         '------------------\n\n'),
        flush=True
        )

## Features ##


if args.load_model is None:

    # features in common between train and user
    overlap_genes = df_train.columns[df_train.columns.isin(df_user.columns)]

    # subset df_train by common genes
    df_train = df_train.loc[:, overlap_genes]
    
    # set best_n_genes
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

    if args.save_model is not None:
        
        # save list of genes of the model
        # you are going to train
        np.save(args.save_model[0], best_n_genes)

        if args.verbose:

            print(('+ Saving genes list of the model '+
                   'into {} +\n\n'.format(args.save_model[0])), 
                  flush = True)

else:
    
    # features in common between mean sd table imported and df_user
    overlap_genes = df_user.columns[
            df_user.columns.isin(pd.read_pickle(args.load_model[1]).index)]

    # set of genes to be considered is the same of
    # the pre-trained model that we are going to load
    best_n_genes = np.load(args.load_model[0], allow_pickle = True)

    # the genes in df_user must include all genes of pre-trained model
    if all(e not in df_user.columns.tolist() for e in best_n_genes.tolist()):
        raise ValueError(
                ('At least one gene of the loaded pre-trained model ' +
                 'is not in the testing dataset.\n' + 
                 'Testing set must contain all genes of the pre-trained model'))

    if args.verbose:

        print(('+ Loaded genes list of the pre-trained model: '+
               '{} +\n\n'.format(args.load_model[0])), 
              flush = True)

# subset df_user by common genes
df_user = df_user.loc[:, overlap_genes] 


## Normalization ##

# Training #

if args.load_model is None: 

    # if user does not load a pre-trained model, 
    # we need to normalize the training matrix

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

    if args.save_model is not None:
    
        # save mean_sd table of pre-trained model
        mean_sd_train_df.to_pickle(args.save_model[1])

        if args.verbose:

            print(('+ Saving mean_sd table of the model '+
                   'into {} + \n\n'.format(args.save_model[1])), 
                  flush = True)

else:

    # load mean and standard deviation of sw-normalized pre-trained model
    mean_sd_train_df = pd.read_pickle(args.load_model[1])

    if args.verbose:

        print(('+ Loaded mean_sd table of pre-trained model: '+
               '{} +\n\n'.format(args.load_model[1])), 
              flush = True)


# Testing #

if args.normalization == 'self':

    if args.verbose:
        print(
            '2. Normalizing data matrix according to "self" method.\n',
            '------------------------------------------------------\n\n',
            'Normalization on user\'s data is independent from training matrix\n',
            flush = True
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
            '2. Normalizing data matrix according to "train" method.\n',
            '-------------------------------------------------------\n\n',
            'The user\'s data is scaled sample-wise,\n',
            'while the gene-wise standardization is performed\n', 
            'using the mean and sd of the sample-wise scaled training matrix\n\n', 
            flush = True)

    # independent sample-wise standardization
    X_user_sw = pd.DataFrame(scale(X = df_user, 
                                   axis = 1), 
                             index = df_user.index, 
                             columns = df_user.columns)
                            
    # Gene-wise standardization

    # transform user_df dataframe using 
    # gene specific mean and standard deviation derived from the training:
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
encoder = joblib.load(encoder_path) 

if args.load_model is None:    

    # subset X_train_scl taking best_n_genes
    # that can either be the overlap between training and testing features
    # or the selected top k genes available in df_user 
    X_train_scl = X_train_scl.loc[:, best_n_genes]

    # load labels (pandas series)
    labels_train = pd.read_pickle(labels_path)
    
    # transform Y labels
    Y_train = encoder.transform(labels_train)    
    
    # train the model
    clf = SVC(kernel = 'linear', C = 1).fit(X_train_scl, Y_train)

    if args.verbose: 
        print(
            '3. Trained classifier on {} genes\n'.format(X_user_scl.shape[1]),
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
        flush = True
        )

    if args.save_model is not None:

        # save pre-trained model
        joblib.dump(clf, args.save_model[2])

        if args.verbose:

            print('+ Saving classifier into {} + \n\n'.format(args.save_model[2]), 
                  flush = True)

else:

    # load pre-trained model
    clf = joblib.load(args.load_model[2])

    if args.verbose:

        print(
            ('3. Loaded pre-trained classifier ' + 
             '{} (n.features: {})\n'.format(args.load_model[2], len(best_n_genes)) + 
             '--------------------------------\n\n'), 
            flush = True
              )


# subset X_user_scl taking best_n_genes
# that can either be
# - the overlap between training and testing features
# - the selected top k genes available in df_user
# - the genes of the loaded pre-trained model 
X_user_scl = X_user_scl.loc[:, best_n_genes]
        

# ---------- #
# Prediction # 
# ---------- #

# Predict cancer types of user's samples

if args.verbose:
    print('4. Predicting the cancer type of the user\'s samples.\n',
            flush = True)

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
        flush = True)
        
        
print("--- %s seconds ---" % (time.time() - start_time))

