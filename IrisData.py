import pandas as pd
import sklearn as sk
import numpy as np
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
iris_data = load_iris()


X=iris_data.data
y=iris_data.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=True)

# SVM MODEL
svm_model= SVC()
svm_model.fit(X_train,y_train)
class_names = ["setosa", "versicolor", "virginica"]  # Replace with your actual class names
label_to_class = {i: name for i, name in enumerate(class_names)}
svm_predictions = svm_model.predict(X_test)
class_predictions = [label_to_class[pred] for pred in svm_predictions]
print(class_predictions)
print(accuracy_score(y_test,svm_predictions))
print(classification_report(y_test,svm_predictions))


#testing with manually given data points
X_new = np.array([  [  5.3, 2.5, 4.6, 1.9 ],[  4.9, 2.2, 3.8, 1.1 ],[3, 2, 1, 0.2]])
manual_prediction= svm_model.predict(X_new)
class_predictions = [label_to_class[pred] for pred in manual_prediction]
print(class_predictions)


# Take user input for new data points
print("Enter new samples for prediction (comma-separated, one sample per line):")
print("Each sample should have 4 numeric values (e.g., 5.3, 2.5, 4.6, 1.9)")
user_input = []

while True:
    line = input("Enter a sample (or type 'done' to finish): ")
    if line.strip().lower() == 'done':
        break
    try:
        sample = list(map(float, line.split(',')))
        if len(sample) == 4:
            user_input.append(sample)
        else:
            print("Each sample must have exactly 4 numeric values. Please try again.")
    except ValueError:
        print("Invalid input. Please enter numeric values separated by commas.")

if user_input:
    X_new = np.array(user_input)
    new_predictions = svm_model.predict(X_new)
    class_predictions = [label_to_class[pred] for pred in new_predictions]
    print(class_predictions)
else:
    print("No samples entered for prediction.")