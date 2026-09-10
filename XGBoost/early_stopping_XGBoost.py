# A splitting the indices of the data into training and test sets using a fixed random seed
train, test = train_test_split(range(len(data)), test_size=0.2, random_state=0)  # A
# B further splitting the training set into training and validation sets using the same random seed
train, validation = train_test_split(train, test_size=0.2, random_state=0)  # B

# C initializing an XGBoost classifier with an early stopping patience for 100 rounds
# D using the ‘error’ parameter, equivalent to accuracy, as an evaluation metric
xgb = XGBClassifier(booster='gbtree',
                    objective='reg:logistic',
                    n_estimators=1000,
                    max_depth=4,
                    min_child_weight=3,
                    early_stopping_rounds=100,  # C
                    eval_metric='error')  # D

X = column_transform.fit_transform(data.iloc[train])
y = target_median[train]

Xv = column_transform.transform(data.iloc[validation])
yv = target_median[validation]

# E fitting the XGBoost classifier to the training data X and labels y and performance on the validation data Xv and yv
xgb.fit(X, y, eval_set=[(Xv, yv)], verbose=False)  # E

Xt = column_transform.transform(data.iloc[test])
yt = target_median[test]

preds = xgb.predict(Xt)
score = accuracy_score(y_true=yt, y_pred=preds)

# F print the accuracy score after comparing the predicted labels with the true labels
print(f"Accuracy: {score:0.5f}")  # F
