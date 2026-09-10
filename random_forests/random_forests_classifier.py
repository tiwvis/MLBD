from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)

# A a RandomForestClassifier with 300 estimators and a minimum number of samples at a leaf node set to 3
model = RandomForestClassifier(n_estimators=300,
                               min_samples_leaf=3,
                               random_state=0)  # A

# B a column transformer that applies different transformations to categorical and numeric features
column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_passthrough, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)  # B

# C a pipeline that sequentially applies column transformation and the random forest classifier model
model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])  # C

# D a five-fold cross-validation using the defined pipeline and calculating accuracy scores
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

# E printing the mean and standard deviation of the accuracy scores from cross-validation
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")  # E
