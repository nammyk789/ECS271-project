import typing
import numpy as np
import numpy as np
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.datasets import make_regression
from tqdm import tqdm_notebook as tqdm
import numpy as np




def GradBoost(model,
              X_test: np.array,                  # testing independent variables
              X_train: np.array,                 # training independent variables
              y_train: np.array,                 # training dependent variable
              boosting_rounds: int = 100,        # number of boosting rounds
              learning_rate: float = 0.1,        # learning rate with default of 0.1
              verbose: bool = True) -> np.array: # if True, shows a tqdm progress bar
    '''
    Takes in a model and performs gradient boosting using that model. This allows for almost any scikit-learn
    model to be used.
    '''

      # make a first guess of our training target variable using the mean
    y_hat_train = np.repeat(np.mean(y_train), len(y_train))
    # initialize the out of sample prediction with the mean of the training target variable
    y_hat_train_test = np.repeat(np.mean(y_train), len(X_test))
    # calculate the residuals from the training data using the first guess
    pseudo_resids = y_train - y_hat_train

    # performs gradient boosting with a tqdm progress bar
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
    return y_hat_train, y_hat_train_test




X, y = make_regression(n_samples=1000,
                       n_features=20,
                       n_informative=15,
                       n_targets=1,
                       bias=0.0,
                       noise=20,
                       shuffle=True,
                       random_state=13)
X_train = X[0:int(len(X) / 2)]
y_train = y[0:int(len(X) / 2)]
X_test = X[int(len(X) / 2):]
y_test = y[int(len(X) / 2):]


#1000 observations and 20 independent variables
#using decisiontree regressor and the ridge regression model for normalization

#print("x train: " + str(X_train))
#print("y train: " + str(y_train))
#print()
#print(X_test)
#print(y_test)

#use decisiontreeclassifier instead for my data- is for multi-class classification, not regression
#the weak learner
tree_model = DecisionTreeRegressor(criterion='squared_error', max_depth=3)

ridge_model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0),
                      fit_intercept=True,
                      normalize=True,
                      cv=3)



##
# Plot the training mean squared error vs. number of boosting rounds by looping through various
# numbers of boosting rounds, calculating the training mean squared error each round and
# appending it to a list.
###


#tqdm is progress bar
tree_mse_train = []
n_rounds = np.arange(5, 101, 5)

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
"""#ridge regression model with cross validation
ridge_mse_train = []
for n_round in tqdm(n_rounds):
    y_hat_train = GradBoost(ridge_model,
                            X_test,
                            X_train,
                            y_train,
                            boosting_rounds=n_round,
                            learning_rate=0.1,
                            verbose=False)[0]

    ridge_mse_train.append(np.mean((y_train - y_hat_train) ** 2))


print(ridge_mse_train)"""



# sets the plot size to 20x8
plt.rcParams['figure.figsize'] = (20,8)

plt.subplot(1, 2, 1)
plt.plot(n_rounds, tree_mse_train)
plt.title('Training MSE vs. Boosting Rounds for Tree Model', fontsize=20)
plt.xlabel('Number of Boosting Rounds', fontsize=15)
plt.ylabel('Training Mean Squared Error', fontsize=15)
plt.show();
