# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 22:00:29 2021

@author: TT
"""

import pandas as pd
import numpy as np
import random

from random import randrange
from random import seed


###########################################################################
def dataClassification(data):
    
    targetFunction_column = data[:, -1]  
    # classfication    
    uniqueClasses, countsUniqueClasses = np.unique(targetFunction_column, return_counts=True)
   
    index = countsUniqueClasses.argmax()
    targetFunc_value = uniqueClasses[index]
    #print(targetFunc_value)
    
    return targetFunc_value

###################################################################################

def find_PotentialSplits(data):
    
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns - 1): # without the last column which is the target function column
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits

#################################################################################

def split(data, split_column, compareValue):
    
    split_column_values = data[:, split_column]

    equalData = data[split_column_values == compareValue]
    unequalData = data[split_column_values != compareValue]
    
    return equalData, unequalData

############################################################################
def entropy(data):
    
    targetFunction_column = data[:, -1] # values of the target function
    _, counts = np.unique(targetFunction_column, return_counts=True)

    probabilities = counts / counts.sum()
    
    entropy = sum(probabilities * -np.log2(probabilities))
    
    return entropy

#############################################################################
def informationGainCalculation(equalData, unequalData):
    
    n = len(equalData) + len(unequalData)
    p_equalData = len(equalData) / n
    p_unequalData = len(unequalData) / n
    
    informationGain =  (p_equalData * entropy(equalData) 
                      + p_unequalData * entropy(unequalData))
   
    return informationGain

############################################################################
# **************  Random Splitting ****************************************#

def random_split(data, potential_splits):
    
    m = -1
    random_split_column = random.choice(list(potential_splits))
    
    while m == random_split_column:
        random_split_column = random.choice(list(potential_splits))
    m = random_split_column
  
    for value in potential_splits[random_split_column]:
       
        return random_split_column, value

##############################################################################
# **************  Information Gain Splitting ********************************#

def informationGainSplit(data, potential_splits):
    
    best_infoGain = 1000
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            equalData, unequalData = split(data, split_column=column_index, compareValue=value)
            current_infoGain = informationGainCalculation(equalData, unequalData)

            if current_infoGain <= best_infoGain:
                best_infoGain = current_infoGain
                best_split_column = column_index
                best_compareValue = value
            
    return best_split_column, best_compareValue

##############################################################################
# **************  Gain Ratio Splitting **************************************#
def gainRatio(data, potential_splits):
    
    best_infoGain = 1000
    max_gain_ratio = -1000
    split_information_entropy = 0
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            equalData, unequalData = split(data, split_column=column_index, compareValue=value)
            current_infoGain = informationGainCalculation(equalData, unequalData)
            
            if current_infoGain <= best_infoGain:
                 best_infoGain = current_infoGain
          
            _, counts = np.unique(potential_splits[column_index], return_counts=True)

            probabilities = counts / counts.sum()
    
            split_information_entropy = sum(probabilities * -np.log2(probabilities))
            
            gainratio = None
            if split_information_entropy != 0:
                gainratio = best_infoGain / split_information_entropy
            else:
                gainratio = -1000
                
            if gainratio > max_gain_ratio:
                max_gain_ratio = gainratio
                best_split_column = column_index
                best_compareValue = value
    
    return best_split_column, best_compareValue
##################################################################################################

def decision_tree_algorithm(df, split_method_choice, counter=0, min_data=2, max_layers=20):
    
    # 1st part
    if counter == 0: # First call of the function
        global COLUMN_NAMES
        COLUMN_NAMES = df.columns
        data = df.values
    else:
        data = df           
    
    
    # Control of the base cases before or at the end of  the recursive part
    if (len(data) < min_data) or (counter == max_layers):
        targetFunc_value = dataClassification(data)
        
        return targetFunc_value

    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = find_PotentialSplits(data)
        if split_method_choice == 1:
            split_column, compareValue = random_split(data, potential_splits)
        elif split_method_choice == 2:
            split_column, compareValue = informationGainSplit(data, potential_splits)
        else:
            split_column, compareValue = gainRatio(data, potential_splits)
            
        equalData, unequalData = split(data, split_column, compareValue)
        
        # Check if there are any of the created dataset are empty
        if len(equalData) == 0 or len(unequalData) == 0:
            targetFunc_value = dataClassification(data)
        
            return targetFunc_value
        
        # Determine question
        attribute_name = COLUMN_NAMES[split_column]
        question = "{} = {}".format(attribute_name, compareValue)
       
        # Instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(equalData, split_method_choice, counter, min_data, max_layers)
        no_answer = decision_tree_algorithm(unequalData, split_method_choice, counter, min_data, max_layers)
        
        sub_tree[question].append(yes_answer)
        sub_tree[question].append(no_answer)
        
        
        return sub_tree
######################################################################################
#************* Function to predict the value of the given examples **********#
def predict_example(example, tree):
    
    # The tree is just a root node
    if not isinstance(tree, dict):
        return tree
    
    # We want to find the question for example "doors=2" to our car?
    question = list(tree.keys())[0] # We take the trees keys (the question into the node) and we convert to a list
    attribute_name, comparison_operator, value = question.split(" ")

    
    if str(example[attribute_name]) == value: # This is the question to compare with the value
        answer = tree[question][0]
    else:
        answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):# If our answer is not a dictionary
        return answer
    
    # recursive part
    else: # If our answer is a dictionary
        residual_tree = answer
        return predict_example(example, residual_tree)

#############################################################################
#********* Predictions for all the examples of a dataframe ***************#

def make_predictions(df, tree):
    
    if len(df) != 0:
        predictions = df.apply(predict_example, args=(tree,), axis=1)
    else:
        predictions = pd.Series()
        
    return predictions

#############################################################################
#************* Accuracy Calculation ***********************##

def calculate_accuracy(df, tree):
    predictions = make_predictions(df, tree)
    predictions_correct = predictions == df['target_function']
    accuracy = predictions_correct.mean()
    
    return accuracy

##############################################################################
################### Evaluation Methods ########################################
#############################################################################
# ************** 1.Function for Holdout Method ****************************#

def holdout_split(df, test_size): # Splits the data into a training and a test set
    
    if isinstance(test_size, float): # If the user gives a float number of test set splitiing
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_dataset = df.loc[test_indices] #Creation of test set
    train_dataset = df.drop(test_indices) #Creation of train set
    
    return train_dataset, test_dataset

##############################################################################
# ********** 2.Function for Κ-Fold Cross Validation Method **************#

def cross_validation_split(dataset, folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size: # While k folds have no yet been created
			index = randrange(len(dataset_copy)) 
			fold.append(dataset_copy.pop(index)) # Create a list with k different folds from the dataset
		dataset_split.append(fold)
	return dataset_split

#############################################################################
####################### MAIN PROGRAM ############################################

col_names = ['buying','maint','doors','persons','lug_boot','safety','target_function']
df = pd.read_csv('car.data', names= col_names)

print("Press 1 for Random Splitting:")
print("Press 2 for Splitting based on Information Gain:")
print("Press 3 for Splitting based on Gain Ratio:")
split_method_choice =int(input("Give your Choice:"))
while split_method_choice!=1 and split_method_choice!=2 and split_method_choice!=3:
    split_method_choice = int(input("Press 1, or 2, or 3 for Splitting Method Choice:"))
if split_method_choice == 1:
    print()
    print("Splitting Choice: Random Splitting")
if split_method_choice == 2:
    print()
    print("Splitting Choice: Information Gain")
if split_method_choice == 3:
    print()
    print("Splitting Choice: Gain Ratio")

############################################################################
################# Holdout Method ###################################
#random.seed(0)
train_dataset, test_dataset = holdout_split(df, test_size=30) 
tree = decision_tree_algorithm(train_dataset, split_method_choice, max_layers=20)
#print(tree)
accuracy = calculate_accuracy(test_dataset, tree)


print()    
print("HOLDOUT-Method")
print("The accuracy with the HOLDOUT-Method is", accuracy)

example = test_dataset.iloc[1]
print("The value of the Target Function is:", test_dataset.iloc[1]['target_function'])
print("Predicted Value is:", predict_example(example, tree))
print()

####################################################################################
############### K_FOLD_CROSS-VALIDATION #############################################

print("10-FOLd Cross Validation Method")
df = df.to_dict('records') # Turn Pandas Dataframe to a Dictionary


folds = cross_validation_split(df, 10)
#for i in range(len(folds)):
    #print("To",i+1," fold einai:",folds[i])
testSet = []
trainSet = []
total_accuracies = []
counter = 0
max_accuracy = -1

for fold in folds:
    counter+=1
    trainSet = list(folds)
    
    trainSet.remove(fold)
    trainSet = sum(trainSet, [])
   
    testSet = list()
    
    for row in fold:
        testSet.append(row)
    #print(testSet)           
    train_set = pd.DataFrame(trainSet)
    #print(train_set)
    tree = decision_tree_algorithm(train_set, split_method_choice, max_layers=20)
            
    test_set = pd.DataFrame(testSet)
    #Calculate Accuracy
    accuracy = calculate_accuracy(test_set, tree)
    total_accuracies.append(accuracy)
    if accuracy> max_accuracy:
        max_accuracy=accuracy

    print(counter, "Fold")
    print("The accuracy for the",counter,"Fold is", accuracy)
print("The best accuracy value was:", max_accuracy)
print()
average_accuracy = sum(total_accuracies) / len(total_accuracies)
print("The average accuracy with the 10-FOLd Cross Validation is", average_accuracy)


example = test_set.iloc[1]
    
print("The value of the Target Function is:", test_set.iloc[1]['target_function'])
print("Predicted Value is:", predict_example(example, tree))

            
   