__authors__ = '1710545'
__group__ = 'Team_05'

from utils_data import read_dataset, read_extended_dataset, crop_images
from KNN import KNN




# FUNCIONS ANÀLISI QUALITATIU
def retrieval_by_color(images, query_colors, color_percentages=None):

    """
    Retorna totes les imatges que contenen les etiquetes de color especificades, ordenades pel percentatge de colors, de més a menys
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
        color_prob: list of K lists, each containing probabilities of the 11 colors sorted in descending order
    """

    matching_images = []
    
    for idx, (img, labels) in enumerate(zip(images, color_labels)):
        # Comprovar si tots els colors de la consulta estan presents
        if (color in labels for color in query_colors):
            if color_percentages is not None:
                # Calcular la puntuació com la suma dels percentatges dels colors buscats
                score = sum(color_percentages[idx].get(color, 0) for color in query_colors)
            else:
                # Si no hi ha percentatges, assignar puntuació 1 (només presència)
                score = 1
            
            matching_images.append((img, score))
    
    # Ordenar per puntuació (de major a menor)
    
    return matching_images.sort(key=lambda x: x[1], reverse=True)



# TEST
def retrieval_by_color_test(K= 3):

    # Inciaialitzar KMeans
    km = KMeans(test_imgs, K, options={'km_init' : 'random', '100': 100})
    # Executar algorisme
    km.fit()

    # Resultats
    centroids = km.centroids
    color_labels, color_probabilities = get_colors2(centroids)

    # Prova funció retrieval_by_color
        # Colors que volem buscar
    query_colors = [color_labels[random.randint(0, len(color_labels)-1)]]
    print(len(test_imgs))
    matching_images = retrieval_by_color(test_imgs, query_colors[0], color_probabilities)

    # Mostrar imatges coincidents
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    for i, img in enumerate(matching_images[:10]):  # Mostra les primeres 10 imatges coincidents
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    return matching_images

retrieval_by_color_test()


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



