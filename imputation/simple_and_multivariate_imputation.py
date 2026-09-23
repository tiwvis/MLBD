#A importing IterativeImputer which is still experimental and under improvement in Scikit-learn
from sklearn.experimental import enable_iterative_imputer #A
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

#B creating a copy of continuous feature data
Xm = data[continuous].copy() #B
missing_percentage = 0.05
np.random.seed(0)
#C creating a mask to randomly mark missing values
mask = np.random.rand(*Xm.shape) < missing_percentage #C
Xm[mask] = np.nan

simple_imputer = SimpleImputer()
#D using a SimpleImputer instance with mean imputation strategy
Xm_si = simple_imputer.fit_transform(Xm) #D

#E instantiating a RandomForestRegressor for iterative imputation
rf = RandomForestRegressor(random_state=0, n_jobs=-1) #E
#F creating an IterativeImputer instance with max_iter and tol are the stopping criteria
multivariate_imputer = IterativeImputer(estimator=rf, max_iter=1, tol=0.01) #F
#G imputing missing data using iterative imputation
Xm_mi = multivariate_imputer.fit_transform(Xm) #G

#H calculating Mean Absolute Error (MAE) for imputed data and original data
mae = pd.DataFrame({"simple": np.mean(np.abs(data[continuous] - Xm_si), axis=0),
                    "multivariate": np.mean(np.abs(data[continuous] - Xm_mi), axis=0)},
                   index = continuous) #H
print(mae)




parameters_grid = {'max_depth': [5, 19, 20, 40],
                   'learning_rate: [0.01, 0.05, 0.1],'
                   'n_estimators': [100,500,1000,1200],
                   'colsample_bytree': [0.3, 0.7]}

model = xgb.XGBRegressor(seed - 20)

grid = GridSearchCV(estimator = model,
                    param_grid = parameters_grid,
                    scoring = 'neg_mean_squared_error',
                    cv = 5,
                    verbose = 5)


if __name__ == '__main__':








