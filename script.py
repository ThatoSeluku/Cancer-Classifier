import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Load dataset
breast_cancer_data = load_breast_cancer()

# Print target values
print(breast_cancer_data.target)

# Print target names
print(breast_cancer_data.target_names)

# Print all feature values (just to inspect the data)
print(breast_cancer_data.data)

# Check what the very first data point is labeled as
first_value = breast_cancer_data.target[0]
print("First data point label:", first_value)
print("Meaning:", breast_cancer_data.target_names[first_value])

# Split dataset
training_data, validation_data, training_labels, validation_labels = train_test_split(
    breast_cancer_data.data,
    breast_cancer_data.target,
    test_size=0.2,
    random_state=100
)

print("Training data length: " + str(len(training_data)))

# Create list for x-axis values (1 to 100)
k_list = list(range(1, 101))
print(k_list)

# Empty list for validation accuracies
accuracies = []

# KNN loop
for k in k_list:
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(training_data, training_labels)

    # Append the accuracy
    accuracies.append(classifier.score(validation_data, validation_labels))

# Print accuracy list
print(accuracies)

# Plot results
plt.plot(k_list, accuracies)
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Validation Accuracy")
plt.title("KNN Accuracy vs Number of Neighbors")
plt.show()
