import enum
import os
import glob
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def euclidean_distances(x, y):
    x_square = np.sum(x*x, axis=1, keepdims=True)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.sum(y*y, axis=1, keepdims=True).T
    distances = np.dot(x, y.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, distances)
    if x is y:
        distances.flat[::distances.shape[0] + 1] = 0.0
    np.sqrt(distances, distances)
    return distances


def Dataloader(data_path, categories, img_num):
    num_categories = len(categories) # number of scene categories.

    train_image_paths = [None] * (num_categories * img_num)
    test_image_paths  = [None] * (num_categories * img_num)

    train_labels = [None] * (num_categories * img_num)
    test_labels  = [None] * (num_categories * img_num)

    for i, label in enumerate(categories):
        images = glob.glob(os.path.join(data_path, 'train', label, '*.jpg'))
        for j in range(img_num):
            train_image_paths[i * img_num + j] = images[j]
            train_labels[i * img_num + j] = label

        images = glob.glob(os.path.join(data_path, 'test', label, '*.jpg'))
        for j in range(img_num):
            test_image_paths[i * img_num + j] = images[j]
            test_labels[i * img_num + j] = label

    return (train_image_paths, test_image_paths, train_labels, test_labels)

def knn_classifier(train_feature, train_labels, test_feature, k=1):
    predict = []
    distances = euclidean_distances(train_feature, test_feature)
    for dist in distances:
        label = []
        idx = np.argsort(dist)
        for i in range(k):
            label.append(train_labels[idx[i]])
        pred = Counter(label).most_common(1)[0][0]
        predict.append(pred)
    return np.array(predict)

def visualize(title, train_labels, test_labels, categories, predicted_categories, save_path):
    # TODO: draw confusion matrix
    num_categories = len(categories)
    train_labels = np.array(train_labels)
    categories = np.array(categories)
    confusion_matrix = np.zeros((num_categories, num_categories))

    for i, label in enumerate(predicted_categories):
        row = np.argwhere(categories == test_labels[i])[0][0]
        col = np.argwhere(categories == predicted_categories[i])[0][0]
        confusion_matrix[row][col] += 1
    
    confusion_matrix /= float(len(test_labels) / num_categories)

    plt.imshow(confusion_matrix, cmap='plasma', interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    locs, labels = plt.xticks()
    plt.xticks(np.arange(num_categories), categories)
    plt.xticks(rotation=90)
    locs, labels = plt.yticks()
    plt.yticks(np.arange(num_categories), categories)
    plt.savefig(save_path, bbox_inches='tight')
