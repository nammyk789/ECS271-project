from random import Random
from decisionTree import *

"""
https://www.youtube.com/watch?v=J4Wdy0Wc_xQ
method:
- randomly sample from data (repeating samples is allowed) until you have a dataset
of the same size
- generate a decision tree, but at each step use a random subset of the features
- do this 100s of times
- for classification, take a vote amongst decision trees
- out of bag samples: samples that were not included in a decision tree
- accuracy: how many out of bag samples are correctly classified by the random forest
- hyperparameter tuning: changing the number of random features we use per branch. start with
the square root of the number of variables
"""
class RandomForest(RandomDecisionTree):
    def __init__(self):
    

def bootstrappedDataset(data):
    """
    get randomly sampled dataset of the same size
    @data: 2D array where each entry is a row in data table
    """
    rand_rows = np.random.randint(low=0, high=len(data), size=len(data))
    data_series = pd.Series(data)
    bootstrapped_data = list(data_series[rand_rows])
    return bootstrapped_data


if __name__ == "__main__":
    df = pd.read_csv("fetal_health.csv")
    np.random.seed(1)
    df.iloc[np.random.permutation(len(df))]