import typing
import numpy as np
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
from statistics import mean
import math



class Grad_Boosting:

    def __init__(self, one_count, zero_count):
        self.one_count = one_count
        self.zero_count = zero_count


    def GradBoostClassifier(self, model,
                  X_test: np.array,                  # testing independent variables
                  X_train: np.array,                 # training independent variables
                  y_train: np.array,                 # training dependent variable
                  boosting_rounds: int = 100,        # number of boosting rounds
                  learning_rate: float = 0.1,        # learning rate with default of 0.1
                  ) -> np.array: # if True, shows a tqdm progress bar
        '''
        Takes in a model and performs gradient boosting using that model. This allows for almost any scikit-learn
        model to be used.
        '''

        #instead of getting the mean of the y training data, instead we calc log(odds) of normal fetus and then
        #pass into logistic function

        log_odds = math.log(self.one_count/self.zero_count)
        print("log odds: " + str(log_odds))

        #now, plug into logistic regression equation
        log_regression = math.exp(log_odds)/(1 + math.exp(log_odds))

        print("logistic regression probability: " + str(log_regression))

        #now we need to calculate residuals (pseudo residuals) for each class - y_train - in our data
        print(y_train[0:10])


        """# calculate the residuals from the training data using the first guess
        pseudo_resids = y_train - y_hat_train
        print("pseudo resids: " + str(pseudo_resids))
        print()"""

        """# performs gradient boosting with a tqdm progress bar
        if verbose:
            from tqdm import tqdm
            # iterates through the boosting round
            for _ in tqdm(range(0, boosting_rounds)):
                # fit the model to the pseudo residuals
                model = model.fit(X_train, pseudo_resids)
                # increment the predicted training y with the pseudo residual * learning rate
                y_hat_train += learning_rate * model.predict(X_train)
                # increment the predicted test y as well
                y_hat_train_test += learning_rate * model.predict(X_test)
                # calculate the pseudo resids for next round
                pseudo_resids = y_train - y_hat_train
        # performs gradient boosting without a progress bar
        else:
            # iterates through the boosting round
            for _ in range(0, boosting_rounds):
                # fit the model to the pseudo residuals
                model = model.fit(X_train, pseudo_resids)
                # increment the predicted training y with the pseudo residual * learning rate
                y_hat_train += learning_rate * model.predict(X_train)
                # increment the predicted test y as well
                y_hat_train_test += learning_rate * model.predict(X_test)
                # calculate the pseudo resids for next round
                pseudo_resids = y_train - y_hat_train

        # return a tuple of the predicted training y and the predicted test y
        return y_hat_train, y_hat_train_test"""



def create_split_and_learner(x, y):

    n_round = 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

    #print(x_train.shape)
    #print(x_test.shape)
    #print(x_train[0:10])
    #print()
    #print(y_train.shape)
    #print(y_test.shape)

    #need to remove header


    #decisiontreeregressor weak learner
    #change squared to 0-1 error
    #change number of leaves?
    tree_model = DecisionTreeRegressor(criterion='squared_error', max_depth=3)
    return x_train, x_test, y_train, y_test, tree_model



def main():

    df = pd.read_csv('fetal_health.csv')
    one_count = 0

    fetal_health_col = df['fetal_health']
    zero_count = 0
    one_count = 0
    n_round = 0
    #two_count = 0
    #three_count = 0
    #num_bins = 4
    #n, bins, patches = plt.hist(fetal_health_col, num_bins, range=[1, 3])
    #plt.show()
    for i in range(0, len(fetal_health_col)):
        if fetal_health_col[i] == 1:
            #trying to make this two class problem for simplicity first
            one_count += 1
        elif fetal_health_col[i] == 2:
            fetal_health_col[i] = 0
            zero_count += 1
        else:
            fetal_health_col[i] = 0
            zero_count += 1

    print("one count: " + str(one_count))
    print()
    print("zero count: " + str(zero_count))
    #print()
    #print(len(fetal_health_col))

    #1655 - cat 1
    #print(one_count)

    #295- cat 2
    #print(two_count)

    #176- cat 3
    #print(three_count)

    #log odds of each event?

    #each point in x_train will be array of features in fetal health dataset

    #we only need the other cols besides fetal health for dataset
    x = df.drop(["fetal_health"], axis = 1)
    y = fetal_health_col

    x_train, x_test, y_train, y_test, tree_model = create_split_and_learner(x, y)
    boost_class = Grad_Boosting(one_count, zero_count)
    boost_class.GradBoostClassifier(tree_model, x_test, x_train, y_train, learning_rate=0.1)




main()


##
# Plot the training mean squared error vs. number of boosting rounds by looping through various
# numbers of boosting rounds, calculating the training mean squared error each round and
# appending it to a list.
###


"""#tqdm is progress bar
tree_mse_train = []
#change to 101 later
n_rounds = np.arange(5, 10, 5)

for n_round in tqdm(n_rounds):
    print("round: " + str(n_round))
    print()
    y_hat_train = GradBoost(tree_model,
                            X_test,
                            X_train,
                            y_train,
                            boosting_rounds=n_round,
                            learning_rate=0.1,
                            verbose=False)[0]

    tree_mse_train.append(np.mean((y_train - y_hat_train) ** 2))

print(tree_mse_train)



# sets the plot size to 20x8
plt.rcParams['figure.figsize'] = (20,8)

plt.subplot(1, 2, 1)
plt.plot(n_rounds, tree_mse_train)
plt.title('Training MSE vs. Boosting Rounds for Tree Model', fontsize=20)
plt.xlabel('Number of Boosting Rounds', fontsize=15)
plt.ylabel('Training Mean Squared Error', fontsize=15)
plt.show();"""
