# A defining a new class NewtonianGradientBoosting as a subclass of GradientBoosting
class NewtonianGradientBoosting(GradientBoosting):  # A
    """the Newton-Raphson method is used to update the predictions"""
    # B setting a regularization parameter reg_lambda
    reg_lambda = 0.25  # B

    # C initializing a constant hessian matrix with ones
    def hessian(self, y_true, y_pred):
        hessian = np.ones_like(y_true)  # C
        return hessian

    def fit(self, X, y):
        self.init = self.logit(np.mean(y))
        y_pred = self.init * np.ones((X.shape[0],))
        # D fitting the decision tree by dividing the negative gradient by
        # the sum of the hessian and the regularization parameter
        for k in range(self.n_estimators):
            gradient = self.gradient(self.logit(y), y_pred)
            hessian = self.hessian(self.logit(y), y_pred)
            tree = DecisionTreeRegressor(**self.params)
            tree.fit(X, -gradient / (hessian + self.reg_lambda))  # D
            self.trees.append(tree)
            y_pred += self.learning_rate * tree.predict(X)


# E creating an instance of the NewtonianGradientBoosting class with specified hyperparameters
cls = NewtonianGradientBoosting(n_estimators=300,
                                learning_rate=0.1,
                                max_depth=4,
                                min_samples_leaf=3,
                                random_state=0)  # E

# F fitting the NewtonianGradientBoosting model to the training data
cls.fit(X, y)  # F
preds = cls.predict(Xt)
score = accuracy_score(y_true=yt, y_pred=preds)
# G predicting target values using the fitted model and calculating the accuracy score for evaluation
print(f"Accuracy: {score:0.5f}")  # G
