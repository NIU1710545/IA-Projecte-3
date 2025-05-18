
import numpy as np
import cv2
import matplotlib.pyplot as plt
from KNN import KNN

def get_shape_accuracy(etiquetes_predites, ground_truth):
    correctes = 0
    total = len(ground_truth)
    for pred, real in zip(etiquetes_predites, ground_truth):
        if pred == real:
            correctes += 1
    return (correctes / total) * 100


def get_shape_accuracy_test(train_imgs_grayscale, train_class_labels, test_imgs, test_class_labels, cropped_images, target_size=(60, 80)):
    print("Executant get_shape_accuracy_test...")
    knn = KNN(train_imgs_grayscale, train_class_labels)
    
    k_values = [1, 3, 5, 7, 9, 11]
    accuracies = []

    for k in k_values:
        predicted_labels = knn.predict(test_imgs, k)
        accuracy = get_shape_accuracy(predicted_labels, test_class_labels)
        accuracies.append(accuracy)
        print(f"Precisió amb k={k}: {accuracy:.2f}%")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Precisió del KNN en funció de K')
    plt.xlabel('Número de veïns (K)')
    plt.ylabel('Precisió (%)')
    plt.grid(True)
    plt.show()

    cropped_resized = []
    for img in cropped_images:
        resized_img = cv2.resize(img, target_size)
        cropped_resized.append(resized_img)

    cropped_resized = np.array(cropped_resized)

    predicted = knn.predict(test_imgs, k=3)


    no_reps = sorted(set(test_class_labels))
    for target_class in no_reps:
        filtered_preds = [p for p, gt in zip(predicted, test_class_labels) if gt == target_class]
        filtered_gts = [gt for gt in test_class_labels if gt == target_class]

        acc = get_shape_accuracy(filtered_preds, filtered_gts)
        print(f"Precisió per a la clase {target_class}: {acc:.2f}%")
