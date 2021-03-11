import numpy as np
import matplotlib.pyplot as plt
import h5py


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0) # set default size of plots
    num_images = len(mislabeled_indices[0])
    col = 15
    row = (num_images//col)+1 
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        plt.subplot(row, col, i + 1)
        #plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3), interpolation='nearest')
        plt.axis('off')
        #plt.title("Prediction: " + classes[int(p[0,index])] + " \n Class: " + classes[y[0,index]])
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
        

def precision_recall (predictions, y_truth, evaluation_set = "Evaluation set"):
    """
    Calclulates precision and recall
    Parameters:
        predictions -- the predicted y-values from dataset
        y_truth -- the truth lables (1/0) from the dataset
    """
    
    #Identify errors
    diff_values = np.squeeze(predictions)- np.squeeze(y_truth)
    true_lingon_list = [true_lingon for true_lingon in np.squeeze(y_truth) if true_lingon == 1]
    true_lingons = len(true_lingon_list)
    
    # Diff value ==1 means false positive i.e. prediction lingon = 1 , but truth is 0 = no-lingon
    #Correct prediction i.e.  True positive = All positives - False positves 
    
    
    all_pred_pos_list = [y_pred for y_pred in np.squeeze(predictions) if y_pred==1]
    false_pred_pos_list =  [y_diff for y_diff in diff_values if y_diff == 1]
    false_pred_neg_list = [y_diff for y_diff in diff_values if y_diff == -1]
    all_pred_pos = len(all_pred_pos_list)      # Number of  predicted as positve i.e. Y_pred = 1
    false_pred_pos = len(false_pred_pos_list)       # Number of falsly predicted positve 
    true_pred_pos = all_pred_pos - false_pred_pos   # Number of correctly predicted positive
    false_pred_neg = len(false_pred_neg_list)       #Number of true positivs falsly predicted negative
    
    #Precision
    precision = true_pred_pos/all_pred_pos  # Share of the ones that were predicted lingon was truely lingon 
    
    # Recall 
    recall = true_pred_pos/(true_pred_pos + false_pred_neg)   # Share of all true lingon that was correctly predicted as lingon
    
    precision_txt = "Precision: {:.0%}"
    recall_txt ="Recall: {:.0%}"
    
    print(evaluation_set + " set evaluation metrics")
    print(precision_txt.format(precision))
    print(recall_txt.format(recall))
    print("Lingon in trainset: ", true_lingons)
    print("Predicted to be lingon: ", all_pred_pos)
    print("Incorrectly predicted to be a lingon:", false_pred_pos)
    print("Correctly predicted to be a lingon: ", true_pred_pos)
    print("Incorrectly predicted not to be a lingon (icke-lingon): ", false_pred_neg)