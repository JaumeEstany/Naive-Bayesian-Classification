import numpy as np

def accuracy(conf_matr, samples_length):
    return float(conf_matr[0][0] + conf_matr[1][1])/samples_length

def precision(conf_matr):
    return float(conf_matr[1][1])/(conf_matr[1][1]+conf_matr[1][0])

def recall(conf_matr):
    return float(conf_matr[1][1]) / (conf_matr[1][1] + conf_matr[0][1])


def confusion_matrix(ground_truth, classification):
    """
    [0][0]: true negative
    [0][1]: false negative
    [1][0]: false positive
    [1][1]: true positive
    """
    conf_matr = [[0, 0], [0, 0]]
    for i in xrange(ground_truth.shape[0]):
        conf_matr[classification[i]][ground_truth[i]] += 1

    return conf_matr

def calculate_all_metrics(ground_truth, classification):

    conf_matr = confusion_matrix(ground_truth, classification)


    accuracy_value = accuracy(conf_matr, ground_truth.shape[0])
    precision_value = precision(conf_matr)
    recall_value = recall(conf_matr)

    return accuracy_value, precision_value, recall_value


def print_all_metrics(resulting_metrics):
    print 'Accuracy: ', resulting_metrics[0]
    print 'Precision: ', resulting_metrics[1]
    print 'Recall: ', resulting_metrics[2]