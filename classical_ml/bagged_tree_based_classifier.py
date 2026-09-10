from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)

# A creating a BaggingClassifier ensemble model based on decision trees
# B setting bootstrap sampling for the BaggingClassifier
# C setting no sampling of features for the BaggingClassifier
# D setting no sampling of data for the BaggingClassifier
model = BaggingClassifier(estimator=DecisionTreeClassifier(),  # A
                          n_estimators=300,
                          bootstrap=True,  # B
                          max_samples=1.0,  # C
                          max_features=1.0,  # D
                          random_state=0)

# E a column transformer that applies different transformations to categorical and numeric features
column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_passthrough, continuous)],
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)  # E

# F a pipeline that sequentially applies column transformation and the bagging classifier model
model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])  # F

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

# G a five-fold cross-validation using the defined pipeline and calculating accuracy scores
# H printing the mean and standard deviation of the accuracy scores from cross-validation
print(f"{mean_cv:0.3f} ({std_cv:0.3f})",
      f"fit: {fit_time:0.2f} secs pred: {score_time:0.2f} secs")  # G
