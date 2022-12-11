import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_matching import match
from homography import solve_homography

def mosaic_stitching(img1, img2, H, is_show=False):
    panorama = cv2.warpPerspective(img1, H, (img1.shape[1]+img2.shape[1], 
                                    img1.shape[0]))
    
    panorama[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    cv2.imwrite('final.png', panorama)
    
    if is_show:
        plt.imshow(panorama[:,:, ::-1])
        plt.show()

    return panorama



if __name__ == '__main__':
    img1 = cv2.imread('../data/data1/112_1300.JPG')
    img2 = cv2.imread('../data/data1/113_1301.JPG')
    kp1, kp2, matches = match(img1, img2, is_default=True)
    H = solve_homography(kp1, kp2, matches, is_default=True)
    print(H)
    panorama = mosaic_stitching(img1, img2, H)
    