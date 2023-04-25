import numpy as np
import math


def evaluate(prediction, true_model, imbalanced_data=False):
    if prediction.shape != true_model.shape:
        return -1
    if len(prediction.shape) != 2:
        return -2
    h, w = prediction.shape
    conf_image = np.zeros((h, w, 3))

    TP, FP, FN, TN = 0, 0, 0, 0
    for r in range(w):
        for c in range(h):
            if prediction[c][r] == true_model[c][r] == 255:
                TP += 1
                conf_image[c][r] = 255, 255, 255
            elif prediction[c][r] == true_model[c][r] == 0:
                TN += 1
                conf_image[c][r] = 0, 0, 0
            elif prediction[c][r] != true_model[c][r] and prediction[c][r] == 255:
                FP += 1
                conf_image[c][r] = 0, 0, 255  # BGR
            elif prediction[c][r] != true_model[c][r] and prediction[c][r] == 0:
                FN += 1
                conf_image[c][r] = 0, 255, 0  # BGR
            else:
                return -3

    accuracy = (TP+TN)/(TP+TN+FN+FP)
    sensitivity = TP/(TP+FN)
    specifity = TN/(FP+TN)
    try:
        precision = TP/(TP+FP)
    except:
        print('No positive values found')
        return np.zeros_like(prediction.shape)
    print('Accuracy: ', accuracy)
    print('Sensitivity: ', sensitivity)
    print('Specifity: ', specifity)
    print('Precision: ', precision)
    if imbalanced_data:
        print('G-Mean: ', math.sqrt(sensitivity*specifity))
        print('F-measure: ', (2*precision*sensitivity)/(precision+sensitivity))
    return conf_image
