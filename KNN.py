__authors__ = ['1707088', '1234567', '7654321']
__group__ = 'aaa'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels, distance_metric='euclidean'):
        self._init_train(train_data)
        self.labels = np.array(labels)
        self.neighbors = None
        self.distance_metric = distance_metric
        
        
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data, feature_type='all'):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        # Convertir a float i reshape a PxD
        train_data = train_data.astype(np.float32)

        # Se'ns proposa mirar de modificar l'espai de característiques
        if feature_type == 'all':
            # tots els pixels
            self.train_data = self.train_data.reshape(self.train_data.shape[0], -1)
        elif feature_type == 'upper_half':
            # Només la meitat superior 
            h = self.train_data.shape[1] // 2
            self.train_data = self.train_data[:, :h, :].reshape(self.train_data.shape[0], -1)
        elif feature_type == 'lower_half':
            h = self.train_data.shape[1] // 2
            self.train_data = self.train_data[:, h:, :].reshape(self.train_data.shape[0], -1)
        elif feature_type == 'mean':
            # Valor mitjà
            self.train_data = np.mean(self.train_data, axis=(1,2))
        elif feature_type == 'std':
            # Desviació estàndard
            self.train_data = np.std(self.train_data, axis=(1,2))

    def get_k_neighbours(self, test_data, k):
        test_data = test_data.astype(np.float32)
        P, M, N = test_data.shape 
        self.test_data = test_data.reshape(P, M * N)
        
        if self.distance_metric == 'euclidean':
            distances = cdist(self.test_data, self.train_data, 'euclidean')
        elif self.distance_metric == 'manhattan':
            distances = cdist(self.test_data, self.train_data, 'cityblock')
        elif self.distance_metric == 'cosine':
            distances = cdist(self.test_data, self.train_data, 'cosine')

        knearest_indices = np.argsort(distances, axis=1)[:, :k]

        # Obtenir ETIQUETES dels veïns
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
