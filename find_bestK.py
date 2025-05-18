import numpy as np
import matplotlib.pyplot as plt

from Kmeans import KMeans 

def test_multiple_llindars(kmeans_instance, max_K, llindars):
    results = {}
    for llindar in llindars:

        best_k = kmeans_instance.find_bestK(max_K, llindar=llindar)
        results[llindar] = best_k
    return results



def find_bestK_function(train_imgs, max_K=10, llindars=None):
    print("Executant find_bestK_test")
    train_imgs = train_imgs[:70]

    if llindars is None:
        llindars = np.arange(5, 35, 5)  # valores por defecto

    kmeans = KMeans(train_imgs, K=3)

    results = test_multiple_llindars(kmeans, max_K, llindars)

    print("Millor K per a cada llindar:", results)

    plt.plot(llindars, list(results.values()), marker='o', linestyle='-', color='b')
    plt.xlabel('Llindar (%)')
    plt.ylabel('K òptima')
    plt.title('Millor K segons el llindar escollit')
    plt.show()
