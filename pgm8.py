from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Load iris dataset
iris_dataset = load_iris()

# Display class names and their corresponding numbers
targets = iris_dataset.target_names
print('Class : Number')
for i in range(len(targets)):
    print(targets[i], " : ", i)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size=0.2, random_state=42)

# Create a KNeighborsClassifier with k=1
kn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier
kn.fit(X_train, Y_train)

# Make predictions on the test set
for i in range(len(X_test)):
    x_new = np.array([X_test[i]])
    prediction = kn.predict(x_new)
    print("Actual: [{0}][{1}], Predicted: {2} {3}".format(Y_test[i], targets[Y_test[i]], prediction[0], targets[prediction[0]]))

# Calculate and print the accuracy of the classifier
accuracy = kn.score(X_test, Y_test)
print("\nAccuracy:", accuracy)
