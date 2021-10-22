# Import any pacakges this module relies on
import pandas as pd


def mean_sd(df): 
    
    """ Returns dataframe with mean and standard deviation for each gene
    
    Parameters
    -----------
    df: numeric dataframe
        Gene expression matrix you want to compute mean and sd
    """
    
    # make dataframe with mean and standard deviation for each gene
    mean_sd_df = pd.concat([df.mean(), df.std()], 
                           axis = 1)
    # assign column names to mean_sd_df
    mean_sd_df.columns = ['mean', 'sd']
    
    return mean_sd_df
