    def Get_color_accuracy(color_labels, ground_truth):
        accuracy = []
    
        for color, ground in zip(color_labels, ground_truth):
            correct_matches = len(set(color) & set(ground))
            total_possible = len(ground)
            accuracy.append(correct_matches / total_possible)
        return accuracy
    
    from collections import defaultdict
