#-------------------------------------------------------------------------
# AUTHOR: your name
# FILENAME: title of the source file
# SPECIFICATION: description of the program
# FOR: CS 4200- Assignment #3
# TIME SPENT: how long it took you to complete the assignment
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTraining = []
dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0
accurate_params = {
    "c" : 0,
    "dgree" : 0,
    "kernel" : "",
    "shape" : ""
}

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here
for c_i in c: #iterates over c
    for degree_i in degree: #iterates over degree
        for kernel_i in kernel: #iterates kernel
           for decision_function_shape_i in decision_function_shape: #iterates over decision_function_shape

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C=c_i, degree=degree_i, kernel=kernel_i, decision_function_shape=decision_function_shape_i)

                #Fit Random Forest to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                accuracte_predictions = 0
                for test in dbTest:
                    class_predicted = clf.predict([test[:-1]])
                    if class_predicted == test[-1]:
                        accuracte_predictions += 1


                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                accuracy = accuracte_predictions / len(dbTest)
                if highestAccuracy < accuracy:
                    highestAccuracy = accuracy
                    accurate_params['c'] = c_i
                    accurate_params['degree'] = degree_i
                    accurate_params['kernel'] = kernel_i
                    accurate_params['shape'] = decision_function_shape_i
                    print(f"Highest SVM accuracy so far: {highestAccuracy},")
                    print(f"Parameters: c={accurate_params['c']}, degree={accurate_params['degree']}, kernel={accurate_params['kernel']}, decision_function_shape={accurate_params['shape']}")

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
#--> add your Python code here
print(f"\nHighest SVM accuracy: {highestAccuracy},")
print(f"Parameters: c={accurate_params['c']}, degree={accurate_params['degree']}, kernel={accurate_params['kernel']}, decision_function_shape={accurate_params['shape']}")