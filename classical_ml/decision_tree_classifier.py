from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import KFold, cross_validate

accuracy = make_scorer(accuracy_score)
cv = KFold(5, shuffle=True, random_state=0)

# A creating a column transformer that applies different transformations to categorical and numeric features
column_transform = ColumnTransformer(
    [('categories', categorical_onehot_encoding, low_card_categorical),
     ('numeric', numeric_passthrough, continuous)],  # A
    remainder='drop',
    verbose_feature_names_out=False,
    sparse_threshold=0.0)

# B an instance of a decision tree classifier
model = DecisionTreeClassifier(random_state=0)  # B

# C a pipeline that sequentially applies column transformation and the decision tree model
model_pipeline = Pipeline(
    [('processing', column_transform),
     ('modeling', model)])  # C

# D a five-fold cross-validation using the defined pipeline, calculating accuracy scores, and returning additional information
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
