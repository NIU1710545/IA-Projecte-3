__authors__ = '1710545'
__group__ = 'Team_05'

from utils_data import read_dataset, read_extended_dataset, crop_images
from KNN import KNN




# FUNCIONS ANÀLISI QUALITATIU
def retrieval_by_color(images, color_labels, query_colors, color_percentages=None):
    matching_images = []
    
    for idx, (img, labels) in enumerate(zip(images, color_labels)):
        # Comprovar si tots els colors de la consulta estan presents
        if all(color in labels for color in query_colors):
            if color_percentages is not None:
                # Calcular la puntuació com la suma dels percentatges dels colors buscats
                score = sum(color_percentages[idx].get(color, 0) for color in query_colors)
            else:
                # Si no hi ha percentatges, assignar puntuació 1 (només presència)
                score = 1
            
            matching_images.append((img, score))
    
    # Ordenar per puntuació (de major a menor)
    matching_images.sort(key=lambda x: x[1], reverse=True)
    
    # Retornar només les imatges (sense les puntuacions)
    return [img for img, score in matching_images]

# FUNCIONS ANÀLISI QUANTITATIU
def kmean_statistics(kmeans, images, Kmax):
    import matplotlib.pyplot as plt
    
    all_pixels = np.concatenate([img.reshape(-1, 3) for img in images], axis=0)


    # Provar diferents valors de K
    K_values = range(1, Kmax+1)
    wcd_values = []
    iterations = []
    
    for K in K_values:
        km = KMeans(all_pixels, K=K, options={'max_iter': 100})
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


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs_grayscale, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    knn = KNN(train_imgs_grayscale, train_class_labels)
    knn.get_k_neighbours([test_imgs[0]], k=2)
    print(knn.neighbors)



