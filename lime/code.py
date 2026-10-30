import lime.lime_tabular
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load the Wisconsin Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # 0 = malignant, 1 = benign (or vice versa depending on dataset convention; check data.target_names)

# 2. Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train the Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model accuracy
accuracy = rf_model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")

# 4. Initialize LIME Tabular Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=data.feature_names,
    class_names=data.target_names,
    mode='classification',
)

# 5. Explain a single prediction (e.g., the first test instance)
i = 0
instance = X_test.iloc[i].values
exp = explainer.explain_instance(
    data_row=instance,
    predict_fn=rf_model.predict_proba,
    num_features=10,  # top 10 features contributing to this prediction
)

# Display explanation in notebook or save/show plot
exp.show_in_notebook(show_table=True, show_all=False)
# Or save as an HTML file:
exp.save_to_file('lime_tumor_explanation.html')
