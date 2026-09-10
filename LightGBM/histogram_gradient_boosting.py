from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)

# A initializing a HistGradientBoostingClassifier with specific hyperparameters for the boosting algorithm
model = HistGradientBoostingClassifier(learning_rate=0.1,
                                       max_iter=300,
                                       max_depth=4,
                                       min_samples_leaf=3,
                                       random_state=0)  # A

# B creating a model pipeline combining data preprocessing (column_transform) and the model
model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])  # B

# C executing five-fold cross-validation on the model pipeline returning scores and trained estimators
cv_scores = cross_validate(estimator=model_pipeline,
                           X=data,
                           y=target_median,
                           scoring=accuracy,
                           cv=cv,
                           return_train_score=True,
                           return_estimator=True)  # C

mean_cv = np.mean(cv_scores['test_score'])
std_cv = np.std(cv_scores['test_score'])
fit_time = np.mean(cv_scores['fit_time'])
score_time = np.mean(cv_scores['score_time'])

# D printing the mean and standard deviation of the accuracy scores from cross-validation
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")  # D
