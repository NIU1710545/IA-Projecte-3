

import numpy as np
import cv2
import matplotlib.pyplot as plt
from KNN import KNN


def retrieval_by_shape(imatges, etiquetes, etiquetes_objectiu, knn_percentatges=None):
    seleccionades = []
    puntuacions = []
    
    for i, (img, forma) in enumerate(zip(imatges,etiquetes)): 
        if forma in etiquetes_objectiu:
            seleccionades.append(img)
            
            if knn_percentatges is not None: 
                puntuacio = sum(knn_percentatges[i].get(f,0) for f in etiquetes_objectiu)
                puntuacions.append(puntuacio)
            else: 
                puntuacions.append(1) 
                
    if knn_percentatges is not None:
        comb = list(zip(seleccionades,puntuacions)) 
        comb.sort(key=lambda x:x[1], reverse =True)
        ordenades = [parella[0] for parella in comb]
        seleccionades = ordenades

    return seleccionades



def retrieval_by_shape_test(cropped_images, class_labels, target_size=(60, 80)):
    print("Executant retrieval_by_shape_test...")
    cropped_resized = [cv2.resize(img, target_size) for img in cropped_images]
    cropped_resized = np.array(cropped_resized)
    
    knn = KNN(cropped_resized, class_labels)
    pred_labels = knn.predict(cropped_resized, k=3)

    no_reps = sorted(set(class_labels))
    for shape_query in no_reps:
        retrieved_imgs = retrieval_by_shape(cropped_resized, pred_labels, [shape_query])
        if len(retrieved_imgs) == 0:
            continue  

        plt.figure(figsize=(10, 4))
        for i, img in enumerate(retrieved_imgs[:10]):
            plt.subplot(2, 5, i + 1)
            plt.imshow(img.astype(np.uint8))
            plt.axis('off')
        plt.suptitle(f"Imatges recuperades per la classe '{shape_query}'")
        plt.tight_layout()
        plt.show()
