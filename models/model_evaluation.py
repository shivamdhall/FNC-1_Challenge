# -*- coding: utf-8 -*-

import numpy as np
import warnings

# filter divide by 0 errors as these are handeled explicitly in the code using np.nan_to_num
warnings.filterwarnings("ignore")

def get_performance_statistics(predictions, labels):
    
    # Get the accuracy of the model as a whole
    total_acc = ((predictions == labels).sum()) / float(labels.size)
    
    # Get precision score fore each class
    agreed_pre = (np.logical_and(predictions == 0, labels == 0).sum()) / float((predictions == 0).sum())
    disagreed_pre = (np.logical_and(predictions == 1, labels == 1).sum()) / float((predictions == 1).sum())
    discuss_pre = (np.logical_and(predictions == 2, labels == 2).sum()) / float((predictions == 2).sum())
    unrelated_pre = (np.logical_and(predictions == 3, labels == 3).sum()) / float((predictions == 3).sum())
    
    # Get recall statistics for each class
    agreed_rec = (np.logical_and(predictions == 0, labels == 0).sum()) / float((labels == 0).sum())
    disagreed_rec = (np.logical_and(predictions == 1, labels == 1).sum()) / float((labels == 1).sum())
    discuss_rec = (np.logical_and(predictions == 2, labels == 2).sum()) / float((labels == 2).sum())
    unrelated_rec = (np.logical_and(predictions == 3, labels == 3).sum()) / float((labels == 3).sum())
    
    # Compute f1 score for each class using precision and recall
    agreed_f1 = 2 * (agreed_pre * agreed_rec) / (agreed_pre + agreed_rec)
    disagreed_f1 = 2 * (disagreed_pre * disagreed_rec) / (disagreed_pre + disagreed_rec)
    discuss_f1 = 2 * (discuss_pre * discuss_rec) / (discuss_pre + discuss_rec)
    unrelated_f1 = 2 * (unrelated_pre * unrelated_rec) / (unrelated_pre + unrelated_rec)

    # In the event that a class is not prediced -- this will lead to division by 0 when calculating precision and f1,
    # this results in nan valus, it is thus necessary to replace thses values with 0
    agreed_f1 = np.nan_to_num(agreed_f1)
    disagreed_f1 = np.nan_to_num(disagreed_f1)
    discuss_f1 = np.nan_to_num(discuss_f1)
    unrelated_f1 = np.nan_to_num(unrelated_f1)
    
    # Get a weightings of each class
    agreed_weight = (labels == 0).sum() / float(labels.size)
    disagreed_weight = (labels == 1).sum() / float(labels.size)
    discuss_weight = (labels == 2).sum() / float(labels.size)
    unrelated_weight = (labels == 3).sum() / float(labels.size)
    
    # Get the weighted mean of the class f1 scores
    weighted_f1 = (agreed_f1*agreed_weight)+(disagreed_f1*disagreed_weight)+(discuss_f1*discuss_weight)+(unrelated_f1*unrelated_weight)
        
    # Get normalised confusion matrix
    confusion_matrix_results = []
    classes = [0,1,2,3]
    for label in classes:
        for predicted in classes:
            res = (predictions[labels == label] == predicted).sum()
            confusion_matrix_results.append(res)
            
    # Reshape to get a confusion matrix
    confusion_matrix = np.array(confusion_matrix_results)
    confusion_matrix = confusion_matrix.reshape(4,4)
    # Normalise the matrix
    confusion_matrix = confusion_matrix / confusion_matrix.sum(axis=1).astype(np.float32)[:,None]
        
    return total_acc, weighted_f1, confusion_matrix