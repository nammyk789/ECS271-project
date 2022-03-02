import typing
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression
from sklearn import tree
from tqdm import tqdm_notebook as tqdm
import numpy as np
import pandas as pd
from statistics import mean
import math



class Grad_Boosting:

    def __init__(self, one_count, zero_count):
        self.one_count = one_count
        self.zero_count = zero_count
        self.first_tree = True


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

        y_pred = log_regression
        #now we need to calculate residuals (pseudo residuals) for each class - y_train - in our data
        #subtract y_train - y_pred for each element in the training set
        #do this for all nums in column and save
        pseudo_resid = [0]*len(y_train)
        count = 0

        for val in y_train:
            pseudo_resid[count] = val - y_pred
            count += 1

        #print(y_train[0:10])
        #print()
        #print(X_train[0:10])
        print("the pseudo resids: " + str(pseudo_resid[0:10]))
        #saved all the pseudo resids


        #fit the decisiontree regressor to the x_train data and pseudo residuals
        # we have 21 features in this data
        model = model.fit(X_train, pseudo_resid)
        tree.plot_tree(model)


        leaf_output_list = self.apply_residual_transformation(model, X_train, y_pred)

        #predict the new probability
        pred_list = []
        pred_list = self.predict_prob(model, X_train, y_pred, leaf_output_list, learning_rate)

        #get new pseudo residuals
        #y_train - pred_list
        new_pseudo_resids = []
        new_pseudo_resids = y_train - pred_list
        new_pseudo_resids = new_pseudo_resids.to_numpy()
        print("the new pseudo residuals: " + str(new_pseudo_resids[0:10]))

        #now need to fit them to a new tree and repeat. put new residual calculation in a loop with
        #the weak learner fitting. keep looping until end of boosting rounds
        



        plt.show()
        #use tree traversal to get leaves - maybe not necessary



    def predict_prob(self, model, X_train, y_pred, leaf_output_list, learning_rate):

        #need to loop through training data, do model.apply to get leaf
        #return the output value from the tuple list stored
        pred_list = []
        log_odds_pred = 0
        actual_prob = 0
        count = 0

        #the i value is correct
        print("shape of x train: " + str(X_train.shape))
        print()
        for i, row in X_train.iterrows():
            if count == 0:
                print(i)
                print(row[8])
                print(row[17])
                count += 1

            reshaped_row = row.values.reshape(1, -1)
            #note, may need to change this
            curr_train_leaf = model.apply(reshaped_row)
            #print("the current training leaf: " + str(curr_train_leaf))
            #print()

            #use the current leaf we've gone to in order to determine the output for log(odds)
            for item in leaf_output_list:
                if item[0] == curr_train_leaf[0]:
                    output = item[1]
                    #print("output is: " + str(item[1]))
                    log_odds_pred = y_pred + (learning_rate*output)
                    #print("the log odds new prediction is: " + str(log_odds_pred))

            #convert log odds back into probability
            actual_prob = (math.exp(log_odds_pred)/(1 + math.exp(log_odds_pred)))
            #print("the probability: " + str(actual_prob))
            pred_list.append(actual_prob)

        #print("the entire pred list: " + str(pred_list))
        return pred_list




    #note: can use model.apply to see where leaf maps to
    #apply the transformation of the residual since we're doing a classification
    def apply_residual_transformation(self, model, X_train, y_pred):


        print(len(model.tree_.value))
        print("value: " + str(model.tree_.value[3]))
        #print(model.tree_.value)
        leaf_indices = model.apply(X_train)

        #below gets the unique leaves we have
        unique_leaves = np.unique(leaf_indices)
        #print(unique_leaves)
        curr_numerator = 0
        curr_denominator = 0
        leaf_output_list = []

        for i in range(0, len(unique_leaves)):
            #print(unique_leaves[i])
            curr_leaf_index = unique_leaves[i]
            curr_numerator = sum(model.tree_.value[curr_leaf_index])
            #print("current numerator: " + str(curr_numerator))

            if self.first_tree:
                curr_denominator = y_pred * (1 - y_pred)

            output = curr_numerator/curr_denominator
            #print("output of current leaf: " + str(output))

            leaf_output_list.append((curr_leaf_index, output[0]))

        print("the leaf output list: " + str(leaf_output_list))
        return leaf_output_list






def create_split_and_learner(x, y):

    n_round = 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

    #need to remove header


    #decisiontreeregressor weak learner
    #change squared to 0-1 error
    #change number of leaves to between 8 and 32 later
    tree_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=3, max_leaf_nodes=3)
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
