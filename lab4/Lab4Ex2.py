import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, make_scorer
from sklearn.datasets import fetch_openml

print("Descarcare dataset Cardio...")
cardio = fetch_openml(name='cardiotocography', version=1, as_frame=False)
X, y = cardio.data, cardio.target

y = np.where(y == '1', 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=42)

parameter_grid = {
    'ocsvm__kernel': ['linear', 'rbf', 'poly'],
    'ocsvm__gamma': ['scale', 0.1, 0.5, 1],
    'ocsvm__nu': [0.05, 0.1, 0.15]
}

pipeline = Pipeline([
    ('scaler', StandardScaler()), 
    ('ocsvm', OneClassSVM())
])

print("Incepere GridSearch (poate dura 1-2 minute)...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring=make_scorer(balanced_accuracy_score),
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

y_test_pred = best_model.predict(X_test)
ba_test = balanced_accuracy_score(y_test, y_test_pred)

print("-" * 30)
print("Best parameters found:", best_params)
print(f"Balanced Accuracy on Test Set: {ba_test:.4f}")
print("-" * 30)