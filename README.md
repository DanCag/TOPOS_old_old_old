TOPOS: Tissue-of-Origin Predictor of Onco-Samples
=================================================

A versatile machine-learning classifier based on SVMs to predict the cancer type of primary, metastasis, cell line and circulating tumor cells samples.


Get TOPOS ready
---------------

You need to have Conda installed as a prerequisite.

1. Download TOPOS repository: `git clone https://github.com/DanCag/TOPOS`
4. Go into topos directory: `cd TOPOS`
5. Extract the compressed archive `training.tar.gz` into `./required_data` directory:
```
tar xvf ./required_data/training.tar.gz -C ./required_data/

```
(you should see the new directory `./required_data/training` after doing this)
6. Make TOPOS executable: `sudo chmod +x topos.py`
7. Create the conda environment: `conda env create -f ./topos.yml`
8. Activate the environment `conda activate TOPOS`

If you want to play with the datasets used in the study,
you need to extract the `./playground.tar.gz` compressed archive


Usage
-----

```
./topos.py [-h, --help] [-n, --n_genes] [-s, --save_model] [-l, --load_model] normalization input_matrix output_predictions
```

Required positional parameters:

* **normalization**

Normalization consists of sample-wise + feature-wise standardization.

First, user's data are standardized sample-wise to remove any artificial variation
due to biases in library preparation, sequencing method, batch effects and so on.
Afterwards, the sample-wise scaled matrix is normalized on the features' level
so that each feature has a mean of 0 and a standard deviation of 1 across samples.
These two steps maintain numberical stability, avoid giving more weigth to features with larger variation
and improve the speed of convergence of the optimization algorithm. 

The two possible values are
 - `train` 
 - `self`

In both of them, the first step consists of standardizing the gene expression matrix sample-wise,
enforcing each sample to have _mean_ = 0 and _standard deviation_ = 1.

Afterwards, if `train`, the feature-wise standardization is performed using
the mean and sd of the sample-wise standardized training matrix.<br>
**Please note that TOPOS was run with `train` flag in all analyses of the manuscript**
 
On the other hand, the `self` option scales the user's data independently of the training data.

* **input\_matrix**<br>
Tab-separated file (tsv) with user's data in the following format:

| | | | | |
| :----:  | :----: | :----: | :----: | :----: |
|         | gene_1 | gene_2 | ...    | gene_n | 
| sample1 |
| sample2 |
| ...     | 
| samplen |

Rows are samples and columns are genes (named with Entrez ids).<br>
Columns and samples must be named, so there will be a column and a row index.<br>
Expression values must be provided in TPM.<br>
You can find the input files used in the study in the folder `./playground/datasets/`.
 
* **output\_predictions**<br>
Path where to write the tab-separated file (tsv) with the predictions.<br>
Sample names will be maintained and predictions will be provided in OncoTree codes.<br>

Optional positional parameters:
 
* **-h, --help**<br>
Shows the basic usage and a description of each parameter.

* **-n, --n_genes**<br>
Number of genes to be used in the training and prediction phase on the user's data.

> If, for instance, ```--n_genes``` is 200, TOPOS will select the 200 most informative genes
according to its gene ranking and will use as many of those genes as are present in the user's data.
If the selected number of genes is larger than 494, then TOPOS will select less genes than the desired number
because after 494, features are not ranked one by one but considering progressively larger steps.
For instance, if the user selects 500 features,TOPOS will take the best 494 genes.
If no number is defined, the overlap between training and testing matrix is taken as the number of features.

TOPOS' training is fast (also when considering all 14369 features).<br>
Nonetheless, we decided to implement the option of storing and reusing a specific trained model
to save time when user wants to apply that specific model to many datasets.

* **-s, --save_model**<br>
Allow user saving the model (user has to specify three output paths)<br>
If not None, TOPOS will save:

   * path of array of genes used by the model (file extension must be `.npy`)
   * path of table with mean and standard deviation:
   
     Mean and sd are computed on sample-wise normalized training matrix<br>
     The features correspond to the overlap between training and testing<br>
     at the time the model was trained (file extension must be `.pkl`) 
   * path of trained classifier (file extension must be `.pkl`) 

* **-l, --load_model**<br>
Allow user loading pre-trained model (user has to specify three input paths)<br>
If not None, TOPOS will load:
                          
   * path of array of genes used by the pre-trained model (file extension must be `.npy`)
   * path of table with mean and standard deviation:
   
     Mean and sd are computed on sample-wise normalized training matrix<br>
     The features correspond to the overlap between training and testing used when computing pre-trained model<br>.
     It is needed if normalization mode is set to "train" (file extension must be `.pkl`) 
   * path of pre-trained classifier (file extension must be `.pkl`)

> Be aware that you can only use a pre-trained model if the dataset you are testing
contains the genes the pre-trained model is trained on.


Examples
--------

1. Picking the top 110 features and scaling the user's data in `train` mode.

```
./topos.py --n_genes 110 train ./playground/datasets/met500_testing/met500_testing_tpm.tsv ./P_met500_110-genes_preds.tsv
```

*Runtime*: ~ 0.3 minute


2. Picking all features and scaling the user's data in `self` mode.

```
./topos.py self ./playground/datasets/met500_testing/met500_testing_tpm.tsv ./P_met500_all-genes_self.tsv
```

*Runtime*: ~ 2.5 minutes


3. Picking the top 110 features, scaling the user's data in `train` mode and saving the model.

```
./topos.py --n_genes 110 --save_model ./genes.npy ./mean_sd.pkl ./clf.pkl train ./playground/datasets/ctc/breast-GSE109761_tpm.tsv ./P_breast-GSE109761_110-genes_train.tsv
```
*Runtime*: ~ 0.3 minute


4. Picking the top 110 features, scaling the user's data in `train` mode, loading the model.

```
./topos.py --n_genes 110 --load_model ./genes.npy ./mean_sd.pkl ./clf.pkl train ./playground/datasets/ctc/breast-GSE111065_tpm.tsv ./P_breast-GSE111065_110-genes_train.tsv
```

*Runtime*: ~ 0.1 minute<br><br>

Runtimes are estimated on the following machine:

| | |
| :----: | :----: |
| **OS**     | Ubuntu 20.04.3 LTS |
| **Memory** | 5.5 Gib     |
| **Processor** | Intel® Core™ i5-8500T CPU @ 2.10GHz × 6 |

