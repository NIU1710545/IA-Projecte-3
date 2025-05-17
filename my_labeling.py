__authors__ = '1710545'
__group__ = 'Team_05'

from utils_data import read_dataset, read_extended_dataset, crop_images
from KNN import KNN



# Cas normal 

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

    def retrieval_by_color_test(K = 3):

        imgatges = []

        for img in test_imgs:
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


    retrieval_by_color_test(K = 4)





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




# Amb menú
    def menu():
        while True:
            print("\n--- MENÚ D'AVALUACIÓ ---")
            print("1. Retrieval per color")
            print("2. Estadístiques KMeans")
            print("0. Sortir")
            opcio = input("Selecciona una opció: ")

            if opcio == '1':
                try:
                    K = int(input("Introdueix el valor de K per KMeans (ex: 4): "))
                except ValueError:
                    print("Valor de K no vàlid.")
                    continue
                retrieval_by_color_test(K)
            elif opcio == '2':
                try:
                    Kmax = int(input("Introdueix el valor màxim de K per a les estadístiques (ex: 10): "))
                except ValueError:
                    print("Valor de K màxim no vàlid.")
                    continue
                kmean_statistics(KMeans, train_imgs_grayscale, Kmax)
            elif opcio == '0':
                print("Sortint del menú.")
                break
            else:
                print("Opció no vàlida. Torna-ho a intentar.")

    menu()
