from lightgbm import LGBMClassifier

accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)

# A Initializing an LGBMClassifier with number of estimators, maximum tree depth, and minimum number of child samples
# B Forcing column-wise histogram building
# B creating a model pipeline that includes a column transformation step and an LGBMClassifier step
lgbm = LGBMClassifier(boosting_type='gbdt',  # A
                      n_estimators=300,
                      max_depth=-1,
                      min_child_samples=3,
                      force_col_wise=True,  # B
                      verbosity=0)

# C performing a five-fold cross-validation using the model pipeline using accuracy scoring
model_pipeline = Pipeline(
    [('processing', column_transform),
     ('lightgbm', lgbm)])  # C

# D printing the mean test score and standard deviation of the test scores obtained during cross-validation
cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)  # D

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])
print(f"CV Accuracy {mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")  # E
