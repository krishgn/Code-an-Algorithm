'''
This script takes data from iris.csv file in data folder taken from
https://archive.ics.uci.edu/ml/datasets/Iris
The data set contains 3 classes of 50 instances each,
where each class refers to a type of iris plant.
Features: sepal length, sepal width, petal length, petal width
Target: class (Iris Setosa, Iris Versicolour, Iris Virginica)
The script takes in filename and no of nearest neighbors as user input.
Designs a KNN Classifier from scratch
'''

import csv
import random
import math
import operator
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

class KNN_Classifier:
    def __init__(self, file_name):
        '''
        Initilizing the data featues, target and calls data processing functions
        '''
        self.data = pd.read_csv(file_name)
        self.data.columns = ['sep_length', 'sep_width','pet_length', 'pet_width', 'target']
        self.features = None
        self.target = None
        self.split_features_target()
        self.normalize()
        self.plot_points()

    def split_features_target(self):
        '''
        Sets the features and target columns of dataset
        '''
        self.features = self.data.iloc[:,:-1]
        self.target = self.data.iloc[:,-1]

    def normalize(self):
        '''
        Normalizes the feature values by scaling to [-1,1]
        '''
        feat_min = self.features.min()
        feat_max = self.features.max()
        self.features = (self.features-feat_min) / (feat_max-feat_min)

    def calc_dist(self, row1, row2):
        '''
        Calculates the distance between two points
        '''
    	sum = 0
    	for i in range(len(row1) - 1):
    		sum+= (row1[i] - row2[i])**2
    	return(math.sqrt(sum))

    def get_k_neighbors(self, k):
        '''
        Gets k neighbors based on distance calculated
        '''
    	total = []
    	for i, row in self.data.iterrows():
    		neighbors = []
    		for j in range(self.data.shape[0]):
    			if j != i:
    				dist = self.calc_dist(self.data.iloc[j], row)
    				neighbors.append((j, dist))
    		neighbors.sort(key=lambda tup: tup[1])
    		total.append(neighbors[:k])
    	return total

    def get_label(self, categories):
        '''
        Finds the label from the most frequent values in the neighbor list
        '''
    	cntr = Counter(categories)
    	most_freq = cntr.most_common()
    	if len(most_freq) == 1 or most_freq[0][1] != most_freq[1][1]:
    		return(most_freq[0][0])
    	else:
    		self.get_label(categories[:-1])

    def predict(self, k):
    	neighbors = self.get_k_neighbors(k)
    	predlist = []
    	for i, name in enumerate(neighbors):
    		categories = [self.data.iloc[category[0]]['target'] for category in name]
    		pred = self.get_label(categories)
    		predlist.append(pred)
    	self.accuracy(self.data['target'].values, predlist, k)

    def accuracy(self, actual, predicted, k):
        '''
        Calculates accuracy
        '''
    	correct = 0
    	for i in range(len(actual)):
    		if actual[i] == predicted[i]:
    			correct += 1
    	print("Accuracy  = " + str(int(100*correct / len(actual))) + str('%'))

if __name__ == "__main__":
    file_name = input("Enter file name: ") or 'data/iris.csv'
    knn = KNN_Classifier(file_name)
    knn.normalize()
    k = input("Enter the value of k:") or '10'
    knn.predict(int(k))
