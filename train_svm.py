from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Assume you have your features (supervectors) and corresponding labels
# 'supervectors' is a numpy array of shape (n_samples, n_features)
# 'labels' is a numpy array of shape (n_samples,) containing 0 (without sigmatism) and 1 (with sigmatism)

# Example:
# supervectors = np.array([[...], [...], ...])  # n_samples x n_features
# labels = np.array([0, 1, 0, 1, ...])  # Binary labels (0 or 1)


X_train, X_test, y_train, y_test = train_test_split(supervectors, labels, test_size=0.2, random_state=42)

# Create an SVM classifier with a polynomial kernel
svm_classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=42)  # 'degree' can be tuned

# Train the SVM classifier on the training data
svm_classifier.fit(X_train, y_train)

# Predict the labels for the test data
y_pred = svm_classifier.predict(X_test)

# Evaluate the model by computing the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM classifier: {accuracy:.2f}")