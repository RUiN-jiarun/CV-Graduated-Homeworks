import cv2
import numpy as np
import matplotlib.pyplot as plt

def gen_sift(img):
    # generate SIFT descriptor
    sift = cv2.SIFT_create() 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    kp, des = sift.detectAndCompute(gray, None) 
    # print(des.shape)
    return kp, des

def gen_cpv(img):
    # generate discriptor formed by concatenated pixel values
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    kp = sift.detect(gray, None)
    des = np.array([cv2.getRectSubPix(gray, (9, 9), p.pt).reshape(-1) for p in kp])
    return kp, des


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


def match(img1, img2, mode='sift', k=2, r=0.7, is_default=True, is_show=False):
    if mode == 'sift':
        kp1, des1 = gen_sift(img1)
        kp2, des2 = gen_sift(img2)
    elif mode == 'cpv':
        kp1, des1 = gen_cpv(img1)
        kp2, des2 = gen_cpv(img2)

    if is_default:
        bf = cv2.BFMatcher()
        matches_raw = bf.knnMatch(des1, des2, k)
        # FLANN_INDEX_KDTREE = 1
        # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        # search_params = dict(checks=50)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches_raw = flann.knnMatch(des1, des2, k)
    else:
        matches_raw = knn(des1, des2, k)

    matches = []
    mask = [[0, 0] for i in range(len(matches_raw))]

    for i, (m, n) in enumerate(matches_raw):
        if m.distance < r * n.distance:
            matches.append(m)
            mask[i] = [1, 0]
            
            # print(m)
    draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=mask,
                    flags=0)
    res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_raw, None, **draw_params) 
    cv2.imwrite('../res/match.jpg', res)

    if is_show:
        plt.imshow(res[:,:, ::-1])
        plt.show()
        
    return kp1, kp2, matches


if __name__ == '__main__':
    img1 = cv2.imread('../data/data1/112_1300.JPG')
    img2 = cv2.imread('../data/data1/113_1301.JPG')
    match(img1, img2)