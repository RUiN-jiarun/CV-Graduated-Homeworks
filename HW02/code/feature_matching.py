import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_sift(img):
    # generate SIFT descriptor
    sift = cv2.SIFT_create() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    kp = sift.detect(gray, None) 
    kp, des = sift.compute(gray, kp) 
    # print(des.shape)
    return kp, des

def gen_cpv(img):
    # generate discriptor formed by concatenated pixel values
    pass

def knn(des1, des2, k):
    matches = []
    for i, f in enumerate(des1):
        fmatch = []
        f_rep = f[None,:].repeat(des2.shape[0], 0)
        distance = np.linalg.norm(f_rep - des2, axis=-1)
        idx = np.argsort(distance)
        for n in range(k):
            fmatch.append(cv2.DMatch(_queryIdx=i, _trainIdx=idx[n], _distance=distance[idx[n]]))
        matches.append(fmatch)
    return matches


def match(img1, img2, mode='sift', k=2, r=0.75, is_default=True):
    if mode == 'sift':
        kp1, des1 = gen_sift(img1)
        kp2, des2 = gen_sift(img2)
    elif mode == 'cpv':
        pass

    if is_default:
        bf = cv2.BFMatcher()
        matches_raw = bf.knnMatch(des1, des2, k)
    else:
        matches_raw = knn(des1, des2, k)

    matches = []
    for m, n in matches_raw:
        if m.distance < r * n.distance:
            matches.append(m)
            # print(m)
    
    res = cv2.drawMatches(img1, kp1, img2, kp2, matches, None) 
    plt.imshow(res[:,:, ::-1])
    plt.show()
    return kp1, kp2, matches


if __name__ == '__main__':
    img1 = cv2.imread('../data/data1/112_1300.JPG')
    img2 = cv2.imread('../data/data1/113_1301.JPG')
    match(img1, img2)