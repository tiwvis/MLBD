from sklearn.tree import DecisionTreeRegressor
import numpy as np


class GradientBoosting():
    def __init__(self, learning_rate=0.1, n_estimators=10, **params):
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.params = params
        self.trees = list()

    # A sigmoid function implementation used for probability transformation that converts logits back into probabilities
    def sigmoid(self, x):
        x = np.clip(x, -100, 100)
        return 1 / (1 + np.exp(-x))  # A

    # B logit function implementation used to transform probabilities into logits
    def logit(self, x, eps=1e-6):
        xp = np.clip(x, eps, 1 - eps)
        return np.log(xp / (1 - xp))  # B

    # C calculating the gradient of the loss function (negative log likelihood) with respect to the predictions
    def gradient(self, y_true, y_pred):
        gradient = y_pred - y_true  # C
        return gradient

    def fit(self, X, y):
        # D initializing the model with the logit-transformed mean of the target values
        self.init = self.logit(np.mean(y))  # D
        y_pred = self.init * np.ones((X.shape[0],))
        # E fitting a decision tree regressor to the negative gradient of the log-odds transformed target
        for k in range(self.n_estimators):
            gradient = self.gradient(self.logit(y), y_pred)
            tree = DecisionTreeRegressor(**self.params)
            tree.fit(X, -gradient)  # E
            self.trees.append(tree)
            # F updating the predicted values using the output of the fitted tree,  with a learning rate factor
            y_pred += self.learning_rate * tree.predict(X)  # F

    def predict_proba(self, X):
        y_pred = self.init * np.ones((X.shape[0],))
        for tree in self.trees:
            # G predicting back requires cumulating predictions from all the trees
            y_pred += self.learning_rate * tree.predict(X)  # G
        return self.sigmoid(y_pred)

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)
        return np.where(proba >= threshold, 1, 0)
