__authors__ = '1710545'
__group__ = 'Team_05'

from utils_data import read_dataset, read_extended_dataset, crop_images
from KNN import KNN

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

