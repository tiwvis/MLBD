from xgboost import XGBClassifier

accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)
# A creating an XGBClassifier model with specified hyperparameters, including the booster type
# B The learning objective, equivalent to Scikit-learn’s loss
# C min_child_weight is equivalent as Scikit-learn’s min_samples_leaf
xgb = XGBClassifier(booster='gbtree',  # A
                    objective='reg:logistic',  # B
                    n_estimators=300,
                    max_depth=4,
                    min_child_weight=3)  # C

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('xgboost', xgb)])

cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
# D printing the mean and standard deviation of cross-validated test scores
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")  # D
