import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from feature_matching import match

def ransac(x, y, threshold, max_iters):
    inliers_cnt = 0
    inliers_idx = []
    iters = 0
    while True:
        iters += 1
        cnt = 0
        # randomly choose 4 pairs
        idx = np.random.randint(0, x.shape[0], 4)
        src = x[idx, ...]
        dst = y[idx, ...]
        H = DLT(src, dst)
        for i in range(x.shape[0]):
            if np.linalg.norm(y[i][None,...] - cv2.perspectiveTransform(x[i][None,...],H)) < threshold:
                cnt += 1
        if cnt > inliers_cnt:
            inliers_cnt = cnt
            inliers_idx = idx
        # update iteration times
        ep = 1 - cnt / x.shape[0]
        max_iters = ransac_update_num_iters(0.995, ep, 4, max_iters)
        if iters > max_iters:
            print('Total iters: ', iters)
            break

    # M estimator
    src = x[inliers_idx, ...]
    dst = y[inliers_idx, ...]
    H = DLT(src, dst)
    return H

def ransac_update_num_iters(p, ep, model_points, max_iters):
    p = max(p, 0.)
    p = min(p, 1.)
    ep = max(ep, 0.)
    ep = max(ep, 1.)

    num = max(1.0 - p, 0.0001)
    denom = 1. - pow(1. - ep, model_points)
    if denom < 0.0001:
        return 0

    num = math.log(num)
    denom = math.log(denom)
    if denom >= 0 or -num >= max_iters * (-denom):
        return max_iters
    else:
        return round(num/denom)

def DLT(x, y):
    A = []      # (2N, 9)
    # (0 0 0 x1 y1 1 -x1y2 -y1y2 -y2)
    # (x1 y1 1 0 0 0 -x1x2 -y1x2 -x2)
    for i in range(x.shape[0]):
        A.append([0, 0, 0, x[i,0,0], x[i,0,1], 1, -y[i,0,1] * x[i,0,0], -y[i,0,1] * x[i,0,1], -y[i,0,1]])
        A.append([x[i,0,0], x[i,0,1], 1, 0, 0, 0, -y[i,0,0] * x[i,0,0], -y[i,0,0] * x[i,0,1], -y[i,0,0]])
    u, s, vt = np.linalg.svd(np.array(A))
    H = (vt[-1, :] / vt[-1, -1]).reshape(3, 3)
    return H

def solve_homography(kp1, kp2, matches, threshold=5.0, max_iters=1000, is_default=True):
    kp1_match = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    kp2_match = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    if is_default:
        H, _ = cv2.findHomography(kp1_match, kp2_match, cv2.RANSAC, threshold, max_iters)
    else:
        H = ransac(kp1_match, kp2_match, threshold, max_iters)

    return H

if __name__ == '__main__':
    img1 = cv2.imread('../data/data1/112_1300.JPG')
    img2 = cv2.imread('../data/data1/113_1301.JPG')
    kp1, kp2, matches = match(img1, img2, is_default=True)
    H = solve_homography(kp1, kp2, matches)
    print(H)
    H = solve_homography(kp1, kp2, matches, is_default=False)
    print(H)