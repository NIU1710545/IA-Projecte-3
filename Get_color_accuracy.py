    def Get_color_accuracy(color_labels, ground_truth):
        accuracy = []
    
        for color, ground in zip(color_labels, ground_truth):
            correct_matches = len(set(color) & set(ground))
            total_possible = len(ground)
            accuracy.append(correct_matches / total_possible)
        return accuracy
    
    from collections import defaultdict

    def test_color_accuracy(images, ground_truth, k=11):
        pred_labels = []

        #calcul dels colors de cada imatge
        for img in images:
            pixels = img.reshape(-1, 3)
            kmeans = KMeans(pixels)
            kmeans.find_bestK(k)
            kmeans.fit()
            
            colors_img = get_colors(kmeans.centroids)
            pred_labels.append(colors_img)

        #calcul precisió per color
        aciertos = defaultdict(int)
        total = defaultdict(int)

        for pred, truth in zip(pred_labels, ground_truth):
            for color in truth:
                total[color] += 1
                if color in pred:
                    aciertos[color] += 1

        precision_color = {color: aciertos[color] / total[color] for color in total}
        print("Rendimiento total: ", sum(aciertos.values()) / sum(total.values()))
        return precision_color
