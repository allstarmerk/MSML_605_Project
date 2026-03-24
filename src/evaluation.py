# will take pairs, images, and threshold and produce metrics
import numpy as np
from src.similarity import cosine_similarity, flatten_embedding

def score_pairs(images, pairs):
    #compute the cosine similarity score for every pair
    a = flatten_embedding(images[pairs])
    b = flatten_embedding(images[pairs[:, 1] ])
    return cosine_similarity(a, b) #higher = more similar, lower = less similar?

def apply_threshold(scores, threshold):
   #convert coninous scores to binary decisions 
    return (scores >= threshold).astype(np.int32)

def compute_metrics(predictions, labels):
    #return accuracy, and true positive rate (same and correct match), false positive rate(perdict same person but different), true negative rate ("diff person prediction" and it is diffrent people), 
    # false negative (predicted diffrent people but actually same person)
   #TPR (true positive rate) =  How good it is at catching matches.
    #FPR (false positive rate) = How often it incorrectly matches different people.
    #accuracy = how many pairs it got right overall.
    #TNR (true negative rate) = How good it is at correctly identifying different people
    tp = int(((predictions == 1) & (labels == 1)).sum()) 
    fp = int(((predictions == 1) & (labels == 0)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    #percision =  out of all the pairs it predicted as matches, how many were actually matches.
    

