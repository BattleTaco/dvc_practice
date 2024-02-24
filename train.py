from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
X, y = load_iris(return_X_y=True)

# Train a simple logistic regression model
clf = LogisticRegression(max_iter=200)
clf.fit(X, y)

# Save the model
joblib.dump(clf, 'model.joblib')
