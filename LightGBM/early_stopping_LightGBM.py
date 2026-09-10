from lightgbm import LGBMClassifier, log_evaluation

# A splitting the indices of the data into training and test sets using a fixed random seed
# B further splitting the training set into training and validation sets using the same random seed
train, test = train_test_split(range(len(data)), test_size=0.2, random_state=0)  # A
train, validation = train_test_split(train, test_size=0.2, random_state=0)  # B

# C initializing a LightGBM classifier with number of estimators, max depth and minimum number of child samples
lgbm = LGBMClassifier(boosting_type='gbdt',
                      early_stopping_round=150,
                      n_estimators=1000,
                      max_depth=-1,
                      min_child_samples=3,
                      force_col_wise=True,
                      verbosity=0)  # C

X = column_transform.fit_transform(data.iloc[train])
y = target_median[train]

Xv = column_transform.transform(data.iloc[validation])
yv = target_median[validation]

# D fitting the LightGBM classifier to the training data X and labels y and performance on the validation data Xv and yv
# E setting accuracy as an evaluation metric
# F setting a callback to suppress the evaluation (period=0)
lgbm.fit(X, y, eval_set=[(Xv, yv)],  # D
         eval_metric='accuracy',  # E
         callbacks=[log_evaluation(period=0)])  # F

Xt = column_transform.transform(data.iloc[test])
yt = target_median[test]

preds = lgbm.predict(Xt)
score = accuracy_score(y_true=yt, y_pred=preds)

# G print the accuracy score after comparing the predicted labels with the true labels
print(f"Test accuracy: {score:0.5f}")  # G
