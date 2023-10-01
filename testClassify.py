import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
import matplotlib.pyplot as plt

# Read the Iris dataset
# Note: Adjust the path to your actual file location
df = pd.read_csv('iris.data')

# Split the data into training and test sets
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train a KNN classifier
score_test = []
models = []
for i in range(1, 30):
    classifier = KNeighborsClassifier(n_neighbors=i)
    classifier.fit(X_train, y_train)
    score_test.append(classifier.score(X_test, y_test))
    models.append(classifier)

# Save the best model
best_model = models[np.argmax(score_test)]
pickle.dump(best_model, open('best_iris_model.pkl', 'wb'))

# Save the classifier class
pickle.dump(KNeighborsClassifier, open('classifier.pkl', 'wb'))

# Print the accuracy of the best model
print("Best model accuracy: {:.2f}%".format(max(score_test)*100))

# Optionally, print additional evaluation metrics
y_pred = best_model.predict(X_test)

# -----------
new_y_test_sample = [10,0.2,0.5,1]
# -----------

print("Classification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))


#-----------------------------------------------------

# Optionally, print additional evaluation metrics
y_pred = best_model.predict(X_test)
print("Classification report:\n", classification_report(y_test, y_pred))
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_mat)

# Visualize the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_mat, annot=True, fmt="g", cmap='Blues', 
            xticklabels=best_model.classes_, 
            yticklabels=best_model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
# plt.show()