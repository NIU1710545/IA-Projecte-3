
import numpy as np



# FUNCIONS ANÀLISI QUANTITATIU
def kmean_statistics(kmeans, images, Kmax, iter=100):
    import matplotlib.pyplot as plt
    
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in images], axis=0)


    # Provar diferents valors de K
    K_values = range(2, Kmax+1)
    wcd_values = []
    iterations = []
    
    for K in K_values:
        km = kmeans(all_pixels, K=K, options={'max_iter': int(iter)})
        km.fit()
        wcd = km.withinClassDistance()
        wcd_values.append(wcd)
        iterations.append(km.num_iter)
    
    # Visualització
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(K_values, wcd_values, 'bo-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Within Class Distance (WCD)')
    plt.title('WCD per diferents valors de K')
    
    plt.subplot(1, 2, 2)
    plt.plot(K_values, iterations, 'ro-')
    plt.xlabel('Nombre de clusters (K)')
    plt.ylabel('Iteracions per convergir')
    plt.title('Convergència per diferents valors de K')
    
    plt.tight_layout()
    plt.show()