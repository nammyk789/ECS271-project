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
import warnings



class Grad_Boosting:

    def __init__(self, one_count, zero_count):
        self.one_count = one_count
        self.zero_count = zero_count
        self.first_tree = True
        #create new dict here for leaf node index: predicted probability in current boosting round
        self.new_prob_dict = {}
        self.first_pred = 0
        self.new_pseudo_resids = []
        self.i = 0

    '''
    Takes in a model and performs gradient boosting using that model. This allows for almost any scikit-learn
    model to be used.
    '''

    def GradBoostClassifier(self, model,
                  X_test: np.array,                  # testing independent variables
                  X_train: np.array,                 # training independent variables
                  y_train: np.array,                 # training dependent variable
                  boosting_rounds: int = 100,        # number of boosting rounds
                  learning_rate: float = 0.001,        # learning rate with default of 0.1
                  ) -> np.array: # if True, shows a tqdm progress bar


        #instead of getting the mean of the y training data, instead we calc log(odds) of normal fetus and then
        #pass into logistic function
        log_odds = math.log(self.one_count/self.zero_count)
        print("log odds: " + str(log_odds))

        #now, plug into logistic regression equation
        log_regression = math.exp(log_odds)/(1 + math.exp(log_odds))
        print("logistic regression probability: " + str(log_regression))

        self.first_pred = log_regression
        print("prediction for first boosting round: " + str(self.first_pred))
        #now we need to calculate residuals (pseudo residuals) for each class - y_train - in our data

        pseudo_resid = [0]*len(y_train)
        count = 0

        for val in y_train:
            pseudo_resid[count] = val - self.first_pred
            count += 1

        print("the pseudo resids: " + str(pseudo_resid[0:10]))
        #saved all the pseudo resids



        #fit the decisiontree regressor to the x_train data and pseudo residuals
        # we have 21 features in this data
        model = model.fit(X_train, pseudo_resid)
        tree.plot_tree(model)


        leaf_output_list = self.apply_residual_transformation(model, X_train)
        #predict the new probability
        pred_list = []
        pred_list = self.predict_prob(model, X_train, y_train, leaf_output_list, learning_rate)

        #get new pseudo residuals
        #y_train - pred_list
        X_train_numpy = X_train.to_numpy()
        y_train_numpy = y_train.to_numpy()
        self.new_pseudo_resids = [0]*len(y_train_numpy)


        for i in range(0, len(y_train_numpy)):
            self.new_pseudo_resids[i] = y_train_numpy[i] - pred_list[i]
            """if i < 10:
                print("x train: " + str(X_train_numpy[i]))
                print("y train: " + str(y_train_numpy[i]))
                print("new pred: " + str(pred_list[i]))
                print("the new pseudo_resid: " + str(self.new_pseudo_resids[i]))
                print()"""




        #tree.plot_tree(model)
        #plt.show()

        loss_list = []
        curr_acc = []
        total_preds = len(y_train_numpy)
        #for loop through boosting rounds
        #fit model to new pseudo residuals
        #apply the residual transformation on the model (apply_residual_transformation)
        #predict the probability of each leaf from the outputs for each training point (predict_prob)
        #update the new pseudo residuals (like above)
        #repeat
        for j in range(0, 10):
            model = model.fit(X_train, self.new_pseudo_resids)
            leaf_output_list = self.apply_residual_transformation(model, X_train)
            pred_list = self.predict_prob(model, X_train, y_train, leaf_output_list, learning_rate)

            correct_preds = 0
            for k in range(0, len(y_train_numpy)):
                if pred_list[k] >= 0.5 and y_train_numpy[k] == 1:
                    correct_preds += 1

                else:
                    if pred_list[k] < 0.5 and y_train_numpy == 0:
                        correct_preds += 1


                self.new_pseudo_resids[k] = y_train_numpy[k] - pred_list[k]


            curr_acc = (correct_preds/total_preds)*100
            print("the current training accuracy for epoch " + str(j) + ": " + str(curr_acc))
            print()
            loss_list.append(sum(self.new_pseudo_resids))

            #calculate accuracy for this round (epoch)

            #print("the new pseudo_resids: " + str(self.new_pseudo_resids[0:10]))
            #print()

        #plotting the final tree
        #tree.plot_tree(model)
        #plt.show()

        print("the loss list: " + str(loss_list))
        print()


        return model, pred_list, loss_list



    def predict_prob(self, model, X_train, y_train, leaf_output_list, learning_rate):

        #need to loop through training data, do model.apply to get leaf
        #return the output value from the tuple list stored
        pred_list = []
        log_odds_pred = 0
        actual_prob = 0
        count = 0
        print()
        #the i value is correct
        for i, row in X_train.iterrows():
            if count == 0:

                #print(row[8])
                #print(row[17])
                #print(y_train[i])
                #print("first sample: " + str(row))
                self.i = i
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

                    #only use this calc with y pred if self.first_tree = true
                    if self.first_tree:
                        log_odds_pred = self.first_pred + (learning_rate*output)

                        #if i == self.i:
                            #print("first pred: " + str(self.first_pred))
                            #print("lr: " + str(learning_rate))
                            #print("output: " + str(output))
                            #print()

                    else:
                        curr_prob = self.new_prob_dict[curr_train_leaf[0]]
                        log_odds_pred = curr_prob + (learning_rate)*output
                    #print("the log odds new prediction is: " + str(log_odds_pred))

                    #convert log odds back into probability
                    actual_prob = (math.exp(log_odds_pred)/(1 + math.exp(log_odds_pred)))

                    #if i == self.i:
                        #print("the probability: " + str(actual_prob))

                    #update new prob dict after first round of boosting for this leaf
                    self.new_prob_dict[curr_train_leaf[0]] = actual_prob

            pred_list.append(actual_prob)


        if self.first_tree:
            self.first_tree = False

        #print("the entire pred list: " + str(pred_list))
        return pred_list




    #note: can use model.apply to see where leaf maps to
    #apply the transformation of the residual since we're doing a classification
    def apply_residual_transformation(self, model, X_train):



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
                curr_denominator = self.first_pred * (1 - self.first_pred)
            else:
                #use the probability from the new prob dict in further boosting rounds
                #need to fix this to add up prev prob from previous rounds
                curr_prob = self.new_prob_dict[curr_leaf_index]
                curr_denominator = curr_prob * (1 - curr_prob)

            output = curr_numerator/curr_denominator

            leaf_output_list.append((curr_leaf_index, output[0]))
        #print()
        #print("the leaf output list: " + str(leaf_output_list))
        return leaf_output_list



    #function to run model on test set
    def test_model(self, tree_model, x_test, y_test):

        x_test_numpy = x_test.to_numpy()
        y_test_numpy = y_test.to_numpy()
        #go through test set and predict class output for x_train values
        total_preds = len(y_test)
        curr_acc = 0

        correct_preds = 0
        print("x test numpy shape: " + str(x_test_numpy[0].shape))
        #print(x_test_numpy[0])
        #print("x test numpy shape: " + str(x_test_numpy[0].shape))



        y_test_pred = tree_model.predict(x_test_numpy)

        #round up from 0.5 to verify prediction of 1, round down for 0
        for i in range(0, len(y_test_pred)):

            if y_test_pred[i] < 0.5 and y_test_numpy[i] == 0:
                correct_preds += 1

            else:
                if y_test_pred[i] >= 0.5 and y_test_numpy[i] == 1:
                    correct_preds += 1

        curr_acc = (correct_preds/total_preds)*100
        print("the current accuracy of this tree on test set: " + str(curr_acc))











def create_split_and_learner(x, y):

    n_round = 0
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15)

    #need to remove header


    #decisiontreeregressor weak learner
    #change squared to 0-1 error
    #change number of leaves to between 8 and 32 later
    tree_model = DecisionTreeRegressor(criterion='absolute_error', max_depth=3, max_leaf_nodes=8)
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

    #print("one count: " + str(one_count))
    #print()
    #print("zero count: " + str(zero_count))
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
    tree_model, pred_list, loss_list = boost_class.GradBoostClassifier(tree_model, x_test, x_train, y_train, learning_rate=0.1)

    #run on test set data now
    boost_class.test_model(tree_model, x_test, y_test)


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
