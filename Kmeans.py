__authors__ = '1710545'
__group__ = 'Team_05'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    def _init_X(self, X):
        """
        Initialization of all pixels, sets X as an array of data in vector form (PxD)
        """
        # Convertir a float i assegurar que és un array numpy
        self.X = np.array(X, dtype=float)
        
        # Si té més de 2 dimensions, transformar a matriu 2D (píxels x canals)
        if self.X.ndim > 2:
            self.X = self.X.reshape(-1, self.X.shape[-1])

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0.00000001
        if 'max_iter' not in options:
            options['max_iter'] = 1000
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        method = self.options['km_init'].lower()
        
        if method == 'first':
            # Obtenir punts únics en ordre d'aparició
            _, idx = np.unique(self.X, axis=0, return_index=True)
            unique_X = self.X[np.sort(idx)]
            
            # Si no hi ha suficients punts únics, completar amb punts aleatoris
            if unique_X.shape[0] >= self.K:
                self.centroids = unique_X[:self.K]
            else:
                # Seleccionar punts aleatoris addicionals sense repetició
                remaining_indices = np.setdiff1d(np.arange(self.X.shape[0]), idx)
                needed = self.K - unique_X.shape[0]
                
                if len(remaining_indices) >= needed:
                    additional_indices = np.random.choice(remaining_indices, needed, replace=False)
                else:
                    additional_indices = np.random.choice(self.X.shape[0], needed, replace=True)
                
                self.centroids = np.concatenate([unique_X, self.X[additional_indices]])
        
        elif method == 'random':
            # Seleccionar K punts aleatoris sense repetició
            indices = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[indices]
        
        elif method == 'custom':
            # Inicialització al llarg de la diagonal de l'hipercub
            min_vals = np.min(self.X, axis=0)
            max_vals = np.max(self.X, axis=0)
            
            if self.K == 1:
                self.centroids = (min_vals + max_vals) / 2
            else:
                self.centroids = np.array([
                    min_vals + (max_vals - min_vals) * (i / (self.K - 1))
                    for i in range(self.K)
                ])
        
        else:
            # Per defecte: inicialització aleatòria
            indices = np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids = self.X[indices]
        
        # Inicialitzar old_centroids
        self.old_centroids = np.copy(self.centroids)
            

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        # Calcular distàncies entre tots els punts i tots els centroides
        dist = np.sqrt(((self.X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :])**2).sum(axis=2))
        
        # Assignar a cada punt el centroide més proper
        self.labels = np.argmin(dist, axis=1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        # Guardar els centroides antics
        self.old_centroids = np.copy(self.centroids)
        
        # Calcular nous centroides
        for k in range(self.K):
            # Obtenir punts assignats a aquest clúster
            cluster_points = self.X[self.labels == k]
            
            if cluster_points.shape[0] > 0:  # Evitar divisions per zero
                self.centroids[k] = np.mean(cluster_points, axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        # Calcular la diferència euclidiana entre centroides nous i antics
        differences = np.linalg.norm(self.centroids - self.old_centroids, axis=1)
        
        # Comprovar si totes les diferències són menors que la tolerància
        return np.all(differences < self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """

        # Inicialitzar centroides
        self._init_centroids()
        
        # Inicialitzar comptador d'iteracions
        self.num_iter = 0
        
        # Bucle principal de l'algorisme
        while self.num_iter < self.options['max_iter']:
            # Assignar etiquetes als punts
            self.get_labels()
            
            # Actualitzar centroides
            self.get_centroids()
            
            # Incrementar comptador
            self.num_iter += 1
            
            # Comprovar convergència
            if self.converges():
                return True
        
        return False

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        wcd = 0.0
        
        for k in range(self.K):
            # Obtenir punts assignats a aquest clúster
            cluster_points = self.X[self.labels == k]
            
            if cluster_points.shape[0] > 0:
                # Calcular distància al centroide per a cada punt del clúster
                dist = np.linalg.norm(cluster_points - self.centroids[k], axis=1)
                
                # Sumar quadrats de les distàncies
                wcd += np.sum(dist**2)
        self.wcd = wcd
        return wcd

    def find_bestK(self, max_K):
        """
        sets the best k analysing the results up to 'max_K' clusters
        """
        wcd_values = []
        
        for k in range(2, max_K + 1):
            self.K = k
            
            self.fit()
            
            # Calcular WCD
            self.withinClassDistance()
            wcd_values.append(self.wcd)
        
            # Calcular percentatge de decrement
            decrement = []
            for i in range(0, len(wcd_values)):
                decrement.append(100 * (wcd_values[i]) / wcd_values[i-1])
            
            # Trobar la millor K
            best_K = max_K  # Valor per defecte
            llindar = 20 
            
            for i in range(len(decrement)):
                if i < len(decrement) - 1 and (100 - decrement[i+1]) < llindar:
                    best_K = i + 2
                    break
        
        self.K = best_K
        return best_K


def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    dist = np.sqrt(np.sum((X[:, np.newaxis, :] - C[np.newaxis, :, :])**2, axis=2))
    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """
    # Obtenir probabilitats per a cada color
    proba = utils.get_color_prob(centroids)
    
    # Obtenir índex del color amb màxima probabilitat per a cada centroide
    color_indices = np.argmax(proba, axis=1)
    
    # Convertir índexs a noms de colors
    color_labels = [utils.colors[i] for i in color_indices]

    return list(zip(color_labels, proba))

