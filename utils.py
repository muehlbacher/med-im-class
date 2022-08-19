import numpy as np
import matplotlib.pyplot as plt
from torch import tensor

def classification(y_train: np.array):
    """'PC-3' 0 , 'U-251 MG' 1, 'HeLa' 2, 'A549' 3, 'U-2 OS' 4, 'MCF7' 5, 'HEK 293' 6, 'CACO-2' 7 and 'RT4' 8 .
        Classification Task with 9 classes - label the input correctly"""
    y_class = np.zeros(shape = len(y_train))
    for i, y in enumerate(y_train):
        #print(y[1])
        if y[1].count('PC-3'):
            y_class[i] = 0
        if y[1].count('U-251 MG'):
            y_class[i] = 1
        if y[1].count('HeLa'):
            y_class[i] = 2 
        if y[1].count('A549'):
            y_class[i] = 3 
        if y[1].count('U-2 OS'):
            y_class[i] = 4 
        if y[1].count('MCF7'):
            y_class[i] = 5 
        if y[1].count('HEK 293'):
            y_class[i] = 6 
        if y[1].count('CACO-2'):
            y_class[i] = 7 
        if y[1].count('RT4'):
            y_class[i] = 8
    return y_class

def trans_classification(target):
    """target: np.array, 
    target[0]: file_id, 
    target[1]: cell_line"""

    if target[1] == 'PC-3':
        return 0
    elif target[1] == 'U-251 MG':
        return 1
    elif target[1] == 'HeLa':
        return 2
    elif target[1] == 'A549':
        return 3
    elif target[1] == 'U-2 OS':
        return 4
    elif target[1] == 'MCF7':
        return 5
    elif target[1] == 'HEK 293':
        return 6
    elif target[1] == 'CACO-2':
        return 7
    elif target[1] == 'RT4':
        return 8
    else:
        raise AttributeError("No Valid cell_line")

def lm_classification(target):
    labels_map = {
        'PC-3': 0,
        'U-251 MG':1,
        'HeLa':2,
        'A549':3,
        'U-2 OS':4,
        'MCF7':5,
        'HEK 293':6,
        'CACO-2':7,
        'RT4':8
    }
    
    try:    
        return labels_map[target[1]]
    except KeyError:
        print(f"Key Error: Label is invalid{target}")

def plot(image: tensor):
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    #plt.imshow(image.permute(1, 2, 0))
    #plt.imshow(image)
    plt.show()
    pass
