

from Kmeans import *
from random import random, randint

# FUNCIONS ANÀLISI QUALITATIU
def retrieval_by_color(imatges, query_colors):
    matching_images = []

    for img, color_prob in imatges:
        score = 0
        for color, prob in color_prob:
            if color in query_colors:
                score += sum(prob)
        if score > 0:
            matching_images.append((img, score))

    # Ordenar les imatges per puntuació (de major a menor)
    matching_images.sort(key=lambda x: x[1], reverse=True)

    # Retornar només les imatges ordenades
    return [img for img, _ in matching_images]

# TEST

def retrieval_by_color_test(test, K = 3):

    imgatges = []

    for img in test:
        km = KMeans(img, K, options={'km_init': 'random', '100': 100})
        km.fit()
        centroids = km.centroids
        color_labels = get_colors(centroids)  # Color + Probabilitat de cada color dels centroids
        
        # Juntar color amb la seva probabilitat
        
        # Afegir img i color_prob a la llista
        imgatges.append([img, color_labels])

    # Tipus de porva (0 = Aleatori, 1 = Manual)
    tipus = 0

    if tipus == 0:
        query_colors = [utils.colors[random.randint(0, len(utils.colors)-1)]]
    else:
        query_colors = ['red', 'green', 'blue']  # Colors a buscar

    matching_images = retrieval_by_color(imgatges, query_colors)


    # Visualització 
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    for i, img in enumerate(matching_images[:10]):  # Mostra les primeres 10 imatges coincidents
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.axis('off')
    plt.show()
    return matching_images
