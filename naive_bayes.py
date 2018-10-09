import random
from datetime import datetime
import pandas as pd
from pandas_confusion import ConfusionMatrix

class NaiveBayes:
    def __init__(self, file_name, attributes, target):
        self.file_name = file_name
        self.full_data = []
        self.training_data = []
        self.test_data = []
        self.attributes = attributes
        self.target = target
        self.separated_by_class = {}
        self.class_frequency = {}
        self.class_probability = {}
        self.features = {}
        self.total_training_instances = 0
        self.attrValDict = {}
        self.attrValList = {}
        self.classNumber = {'good' : 0, 'vgood' : 1, 'acc' : 2 , 'unacc' : 3}


    # runs the Naive Bayes Classifier
    def execute(self,flag):
        self.read_file()
        self.readAttrList()
        self.separate_by_class()
        self.count_all_occurances()
        self.calculate_class_frequency()
        self.calculate_class_probability()
        return self.get_instance_to_classify(flag)


    def count_all_occurances(self):

        for every_class in self.separated_by_class: #gives all classes
            #print(len(self.separated_by_class[every_class]))
            for instance in self.separated_by_class[every_class]:
                for attributeIndex, atributeValue in enumerate(instance):
                    if attributeIndex == 6:
                        break
                    #print(attributeIndex)
                    attributeName = self.attributes[attributeIndex]
                    #print(attributeName)
                    self.attrValDict[attributeName][atributeValue][self.classNumber[every_class]] += 1



    def readAttrList(self):
        with open("cardaten/attrList.txt", "r") as file:
            d = {}
            for line in file:
                line = line.strip("\r\n\t")
                key, val = (line.split(':'))
                val_list = val.split(",")
                new_list = {}
                self.attrValList[key] = val_list
                for l in val_list:
                    new_list[l] = [0, 0, 0, 0]
                self.attrValDict[key] = new_list


    def read_file(self):
        with open(self.file_name, "r") as file:
            for line in file:
                line = line.strip("\r\n")
                self.full_data.append(line.split(','))

        # get random indices to use as training data (2/3rd of the file)
        # populate the training data list with the instances in those randomly calculated indices
        # populate the test data list with the remaining instances
        training_data_indices = random.sample(range(0, len(self.full_data)-1), int(2 / 3 * len(self.full_data)))

        for index in training_data_indices:
            # populate the training_data list with the randomly calculated indices
            self.training_data.append(self.full_data[index])

        for instance in self.full_data:
            # add the test data instances to the test data list by checking
            # if the instances are in the training data list; if not, they are test data
            if instance not in self.training_data:
                self.test_data.append(instance)

        self.training_data = self.full_data




    # this function separates the instances by class value
    def separate_by_class(self):
        # loop over the training dataset
        for instance in self.training_data:
            # get the current class value (acc/unacc/good/vgood)
            class_value = instance[-1]
            # if the current class value is not already in the separated dictionary,
            # create a new key with that class value.
            # Add the instance to the key
            if class_value not in self.separated_by_class.keys():
                self.separated_by_class[class_value] = []
            self.separated_by_class[class_value].append(instance)


    # This function calculates the frequency of each class value (unacc: 560, acc: 320... etc)
    def calculate_class_frequency(self):
        for instance in self.training_data:
            class_value = instance[-1]
            # Loop through the training data set to count the occurrences of each class value
            if class_value in self.class_frequency:
                self.class_frequency[class_value] += 1
            else:
                self.class_frequency[class_value] = 1


    # This function divides the count of a class value by the total number of instances
    # to calculate the probabilities of each class value
    def calculate_class_probability(self):
        # Calculate the total number of training instances
        for key in self.class_frequency:
            self.total_training_instances += self.class_frequency[key]
        # Calculate each class' probability by dividing the count of that class by the total training instances
        full_pro = 0
        for val in self.class_frequency:
            self.class_probability[val] = self.class_frequency[val] / self.total_training_instances
            full_pro = full_pro + self.class_probability[val]




    def classify(self,test_instance):

        start = datetime.now()

        testAttributeValues = test_instance
        laplace_correction = 0
        assigned_class = ""
        probability_assigned_class = 0
        m = 1

        for every_class in self.separated_by_class:
            # set flag for correction
            for attributeIndex, atributeValue in enumerate(testAttributeValues):
                attributeName = self.attributes[attributeIndex]
                if (self.attrValDict[attributeName][atributeValue][self.classNumber[every_class]]) == 0:
                    laplace_correction = 1  # do correction
                else:
                    next
            p = self.class_probability[every_class]

            for attributeIndex, atributeValue in enumerate(testAttributeValues):
                attributeName = self.attributes[attributeIndex]
                if not laplace_correction:
                    p *= (self.attrValDict[attributeName][atributeValue][self.classNumber[every_class]]
                          / self.class_frequency[every_class])
                else:

                    p *= (((self.attrValDict[attributeName][atributeValue][self.classNumber[every_class]])
                          + m * (1/len(self.attrValList[attributeName])) )/ (self.class_frequency[every_class]+ m))

            if p > probability_assigned_class:
                probability_assigned_class = p
                assigned_class = every_class


        return assigned_class


    def get_instance_to_classify(self,flag):

        actual = []
        predicted = []

        correct = 0
        error = 0

        for instance in  self.test_data:
            classification = self.classify(instance[0:len(instance)-1])
            actual.append(instance[-1])
            predicted.append(classification)

            if classification == instance[-1]:
                correct += 1
            else:
                error += 1

        print("correct : ", correct, "error : ", error)

        if flag:

            y_actu = pd.Series(actual, name='Actual')
            y_pred = pd.Series(predicted, name='Predicted')
            df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
            print("Confusion Matrix for the 100th Sample:")
            print(df_confusion)

        return error




