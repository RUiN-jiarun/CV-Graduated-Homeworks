import numpy as np
import cv2
from utils import Dataloader, euclidean_distances, knn_classifier, visualize
import os
from PIL import Image

def calcSIFT(img, stride=10, size=16):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp = [cv2.KeyPoint(x, y, stride) for y in range(int(size/2), img.shape[0], stride) 
                                    for x in range(int(size/2), img.shape[1], stride)]
    # kp, des = sift.detectAndCompute(gray, None)
    _, des = sift.compute(img, kp)
    return des

def build_vocabulary(dir, vocab_size=150):
    bag_of_features = []
    for img_path in dir:
        img = cv2.imread(img_path)
        des = calcSIFT(img)
        bag_of_features.append(des)
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')

    criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
    compactness, labels, centers = cv2.kmeans(bag_of_features, vocab_size, 
                                                bestLabels=None, 
                                                criteria=criteria, 
                                                attempts=20, 
                                                flags=cv2.KMEANS_PP_CENTERS)
    vocab = np.vstack(centers)
    return vocab

def get_bag_of_sifts(dir):
    vocab = np.load('vocab.npy')
    image_feats = []
    for img_path in dir:
        img = cv2.imread(img_path)
        des = calcSIFT(img)
        dist = euclidean_distances(vocab, des)
        min_dist_idx = np.argmin(dist, axis=0)
        hist, bin_edges = np.histogram(min_dist_idx, bins=len(vocab))
        hist_norm = hist / np.linalg.norm(hist)

        image_feats.append(hist_norm)
    return np.array(image_feats)


def main():
    categories = ['Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial',
           'InsideCity', 'Kitchen', 'LivingRoom', 'Mountain', 'Office',
           'OpenCountry', 'Store', 'Street', 'Suburb', 'TallBuilding']
    train_image_paths, test_image_paths, train_labels, test_labels = Dataloader('../data', categories, img_num=100)
    if not os.path.isfile('vocab.npy'):
        print('Building vocabulary...')
        vocab = build_vocabulary(train_image_paths, vocab_size=200)
        np.save('vocab.npy', vocab)
        print('Done.')
    print('Start matching...')
    train_feature = get_bag_of_sifts(train_image_paths)
    test_feature = get_bag_of_sifts(test_image_paths)
    result = knn_classifier(train_feature, train_labels, test_feature, k=1)
    accuracy = float(len([x for x in zip(test_labels, result) if x[0]== x[1]]))/float(len(test_labels))
    print('Average accuracy: ' + str(accuracy))
    for category in categories:
        accuracy_each = float(len([x for x in zip(test_labels, result) if x[0]==x[1] and x[0]==category]))/float(test_labels.count(category))
        print(str(category) + ': ' + str(accuracy_each))
    visualize('Bag of Words & KNN', train_labels, test_labels, categories, result, '../results/bow_knn')

if __name__ == '__main__':
    main()
