TOPOS: Tissue-of-Origin Predictor of Onco-Samples
=================================================

A versatile machine-learning classifier based on SVMs to predict the cancer type of primary, metastasis, cell line and circulating tumor cells samples.

Installation
------------

If you don't have the required python3 modules installed (pandas, scikit-learn and scipy), go to step 1.<br>
If you have them installed, you can go directly to step 2.

### 1. Installing required python modules

Commands that you need to execute in order to get Ubuntu 20.04 LTS ready after a fresh installation.

```
sudo apt install python3-pip
pip3 install pandas
pip3 install scikit-learn==0.23.2
pip3 install scipy
pip3 install numpy==1.19.1
```

### 2. Getting TOPOS ready

1. Download TOPOS repository<br>
2. Unzip the file ```unzip TOPOS-main.zip```

If you want to play with the datasets used in the study,<br>
you need to separately downloaded the playground.tar.gz from Github


Usage
-----

```
./topos [-h] [-v VERBOSE] [-n, --n_genes] normalization input_matrix output_predictions
```

Required positional parameters:

* ***normalization***: strategy used to scale user's data.<br>

> Normalization consists of sample-wise + feature-wise standardization.<br>
First, user's data are scaled sample-wise to remove any artificial variation<br>
due to biases in library preparation, sequencing method, batch effects and so on.<br>
Afterwards, the sample-wise scaled matrix is normalized on the features' level<br>
so that each feature has a mean of 0 and a standard deviation of 1 across samples.<br>
This second step maintains numberical stability, avoids giving more weigth to features with larger variation<br>
and improves the speed of convergence of the optimization algorithm.<br> 

The two possible values are
 - ``` train ``` 
 - ``` self ```<br>

In both of them, the first step consists of standardizing the gene expression matrix sample-wise,<br>
enforcing each sample to have _mean_ = 0 and _standard deviation_ = 1.<br>

Afterwards, if ``` train ```, the feature-wise standardization is performed using<br>
the mean and sd of the sample-wise standardized training matrix.<br>
 
On the other hand, the ``` self ``` option scale the user's data independently of the training data.

* ***input\_matrix***: tab-separated file (tsv) with user's data in the following format:<br>
rows are samples and columns are genes (named with Entrez ids).<br>
Columns and samples must be named, so there will be a column and a row index.<br>
Expression values must be provided in TPM.<br>
You can find input files used in the study in the folder ``` ./playground/datasets/ ```.<br>
 
* ***output\_predictions***: path where to write the tab-separated file (tsv) with the predictions.<br>
Sample names will be maintained and predictions will be provided in OncoTree codes.<br>
 
 
Optional positional parameters:
 
* ***-h, --help***: shows the basic usage and a description of each parameter.
 
* ***-v, --verbose***: control the verbosity of execution.<br>
If ``` True ```, an explanation for each step performed will be printed to ``` stdout ```.

* ***-n, --n_genes*** : number of genes to be used in the training and prediction on the user's data.<br>
For example, if ```--n_genes``` is 200, TOPOS will select the 200 most informative genes according to its gene ranking ,<br> 
and will use as many of those genes as are present in the user's data. <br>
If the selected number of genes is larger than 494, then TOPOS will select less genes than the desired number because<br>
after 494 features are not ranked one by one but considering progressively larger steps. For instance,<br>
if the user selects 500 features,TOPOS will take the best 494 genes.<br>
If no number is defined, the overlap between training and testing matrix is taken as the number of features.<br>
We recommend to set ```--n_genes``` lower than 500 so that TOPOS' execution time stays short.<br>

Example
-------

Picking the top 110 features, scaling the user's data in ```train``` mode and saving the predictions to `P_met500_110-genes.tsv`.

```
./topos.py --verbose True --n_genes 110 train ./playground/datasets/prim-met-lines/met500/met500_testing_tpm.tsv ./executable_met500_110-genes_preds.tsv
```
