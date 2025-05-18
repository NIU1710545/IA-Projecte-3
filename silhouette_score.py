from Kmeans import KMeans 
from sklearn.metrics import silhouette_score
import numpy as np




def test_silhouette_scores(train_imgs, max_K=10):
    X = np.array([img.flatten() for img in train_imgs])
    
    scores = {}
    for k in range(2, max_K + 1):  
        print(f"\nProvant K={k}")
        kmeans = KMeans(X, K=k)
        converged = kmeans.fit()  
        
        labels = kmeans.labels  
        
        n_clusters_found = len(set(labels))

        
        if n_clusters_found > 1:
            score = silhouette_score(X, labels)
            print(f"Silhouette Score per a K={k}: {score:.4f}")
            scores[k] = score

    
    return scores

