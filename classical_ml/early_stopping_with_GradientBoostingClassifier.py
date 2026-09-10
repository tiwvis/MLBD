#A a GradientBoostingClassifier model whose iterations are raised to 1000 from the previous 300
#B as a validation fraction, the GradientBoostingClassifier uses 20% of the training data for validation
#C the training of the GradientBoostingClassifier will stop after 10 iterations without improvements on the validation
model = GradientBoostingClassifier(n_estimators=1000, #A
                                   learning_rate=0.1,
                                   validation_fraction=0.2, #B
                                   n_iter_no_change=10, #C
                                   max_depth=4,
                                   min_samples_leaf=3,
                                   random_state=0)

model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])

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
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")
#D extracting the number of estimators used during training for each fold's estimator
iters = [cv_scores["estimator"][i].named_steps["modeling"].n_estimators_
         for i in range(5)]  #D
#E printing the list of the number of estimators for each fold's estimator
print(iters) #E