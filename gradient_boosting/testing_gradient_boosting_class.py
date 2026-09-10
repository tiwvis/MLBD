from sklearn.model_selection import train_test_split

# A splitting the dataset indices into training and test sets using a fixed random seed
train, test = train_test_split(range(len(data)), test_size=0.2, random_state=0)  # A

# B initializing a GradientBoosting model with specified hyperparameters
cls = GradientBoosting(n_estimators=300,
                       learning_rate=0.1,
                       max_depth=4,
                       min_samples_leaf=3,
                       random_state=0)  # B

# C applying the column transformations to the training data
# D extracting the target values corresponding to the training data
X = column_transform.fit_transform(data.iloc[train])  # C
y = target_median[train]  # D

cls.fit(X, y)

# E applying the same column transformations to the test data
Xt = column_transform.transform(data.iloc[test])  # E

# F extracting the target values corresponding to the test data
yt = target_median[test]  # F

preds = cls.predict(Xt)

# G calculating the accuracy score by comparing the predicted labels with the actual test labels
score = accuracy_score(y_true=yt, y_pred=preds)  # G

# H printing the calculated accuracy score
print(f"Accuracy: {score:0.5f}")  # H
