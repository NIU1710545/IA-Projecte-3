__authors__ = '1710545'
__group__ = 'Team_05'

from utils_data import read_dataset, read_extended_dataset, crop_images
from KNN import KNN
from Kmeans import *
from retrieval_by_color import *
from kmean_statistics import kmean_statistics


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
                    K = int(input("Introdueix el valor de2 K per KMeans (ex: 4): "))
                except ValueError:
                    print("Valor de K no vàlid.")
                    continue
                retrieval_by_color_test(test_imgs, K )
            elif opcio == '2':
                try:
                    Kmax = int(input("Introdueix el valor màxim de K per a les estadístiques (ex: 10): "))
                except ValueError:
                    print("Valor de K màxim no vàlid.")
                    continue
                kmean_statistics(KMeans, train_imgs_grayscale, Kmax, 100)  
            elif opcio == '0':
                print("Sortint del menú.")
                break
            else:
                print("Opció no vàlida. Torna-ho a intentar.")

    menu()
