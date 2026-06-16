
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Topic 3: Securing a Model Against Attack")
print("Using teacher-student approach inspired by PATE")

# Load digit image dataset
digits = load_digits()
X = digits.data
y = digits.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create teacher models
num_teachers = 5
teacher_models = []

X_splits = np.array_split(X_train, num_teachers)
y_splits = np.array_split(y_train, num_teachers)

print("\nTraining teacher models...")

for i in range(num_teachers):
    teacher = LogisticRegression(max_iter=2000)
    teacher.fit(X_splits[i], y_splits[i])
    teacher_models.append(teacher)
    print(f"Teacher {i+1} trained on {len(X_splits[i])} samples")

# Aggregate teacher predictions
print("\nAggregating teacher predictions...")

teacher_predictions = []

for teacher in teacher_models:
    pred = teacher.predict(X_test)
    teacher_predictions.append(pred)

teacher_predictions = np.array(teacher_predictions)

# Majority vote
aggregated_labels = []

for i in range(teacher_predictions.shape[1]):
    votes = teacher_predictions[:, i]
    counts = np.bincount(votes, minlength=10)
    final_label = np.argmax(counts)
    aggregated_labels.append(final_label)

aggregated_labels = np.array(aggregated_labels)

teacher_accuracy = accuracy_score(y_test, aggregated_labels)

print("\nAggregated Teacher Accuracy:", teacher_accuracy)

# Add noise for privacy protection
print("\nAdding noise to teacher votes for privacy protection...")

noise_scale = 1.0
private_labels = []

for i in range(teacher_predictions.shape[1]):
    votes = teacher_predictions[:, i]
    counts = np.bincount(votes, minlength=10)

    noisy_counts = counts + np.random.laplace(
        loc=0,
        scale=noise_scale,
        size=counts.shape
    )

    private_label = np.argmax(noisy_counts)
    private_labels.append(private_label)

private_labels = np.array(private_labels)

private_accuracy = accuracy_score(y_test, private_labels)

print("Private Aggregated Teacher Accuracy:", private_accuracy)

# Train student model using private labels
print("\nTraining student model using private labels...")

student_model = LogisticRegression(max_iter=2000)
student_model.fit(X_test, private_labels)

student_predictions = student_model.predict(X_test)
student_accuracy = accuracy_score(y_test, student_predictions)

print("Student Model Accuracy:", student_accuracy)

print("\nClassification Report:")
print(classification_report(y_test, student_predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, student_predictions))

# Privacy explanation
epsilon_estimate = 1 / noise_scale

print("\nPrivacy Analysis")
print("Noise Scale:", noise_scale)
print("Estimated epsilon:", epsilon_estimate)
print("Higher noise gives stronger privacy but may reduce accuracy.")
print("Lower noise gives weaker privacy but may improve accuracy.")
