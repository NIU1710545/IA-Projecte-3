__authors__ = ['1707088', '1234567', '7654321']
__group__ = 'aaa'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.neighbors = None
        
        
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        train_data = train_data.astype(np.float32)
        P, M, N = train_data.shape 
        self.train_data = train_data.reshape(P, M * N)

    def get_k_neighbours(self, test_data, k):
        test_data = test_data.astype(np.float32)
        P, M, N = test_data.shape 
        self.test_data = test_data.reshape(P, M * N)

        distances = cdist(self.test_data, self.train_data, metric='euclidean')
        knearest_indices = np.argsort(distances, axis=1)[:, :k]

        self.neighbors = self.labels[knearest_indices]
        
    
    def get_class(self):

        predicted_classes = []

        for neighbor_row in self.neighbors:
            class_count = {}

            for label in neighbor_row:
                class_count[label] = class_count.get(label, 0) + 1
            predicted_class = max(class_count, key=class_count.get)
        
            predicted_classes.append([predicted_class])

        return np.ravel(predicted_classes)


    def predict(self, test_data, k):

        self.get_k_neighbours(test_data, k)
        predicted_classes = self.get_class()
    
        return predicted_classes
