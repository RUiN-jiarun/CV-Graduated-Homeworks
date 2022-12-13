import cv2
import numpy as np
import matplotlib.pyplot as plt
from feature_matching import match
from homography import solve_homography
import glob

def read_img_list(dir):
    # dir = '../data/data2/'
    image_lst = glob.glob(dir + '*.JPG')

    print(image_lst)

    nums = len(image_lst)
    images = [cv2.imread(f) for f in image_lst]
    
    center = nums / 2
    print("center index: %d" % center)

    left_list = []
    right_list = []
    for i in range(nums):
        if (i <= center):
            left_list.append(images[i])
        else:
            right_list.append(images[i])
    # left_list = images

    return left_list, right_list

def left_shift(left_list, right_list):
    # left_list = list(reversed(left_list))
    img_pre = left_list[0]
    for img_append in left_list[1:]:
        H = match_H(img_pre, img_append)
        print(H)
        Ht = np.linalg.inv(H)

        bottom_right = np.dot(Ht, np.array([img_pre.shape[1], img_pre.shape[0], 1]))
        top_left = np.dot(Ht, np.array([0, 0, 1]))
        bottom_left = np.dot(Ht, np.array([0, img_pre.shape[0], 1]))
        top_right = np.dot(Ht, np.array([img_pre.shape[1], 0, 1]))
        
        bottom_right /= bottom_right[-1]
        top_left /= top_left[-1]
        bottom_left /= bottom_left[-1]
        top_right /= top_right[-1]

        cx = int(max([0, img_pre.shape[1], top_left[0], bottom_left[0], top_right[0], bottom_right[0]]))
        cy = int(max([0, img_pre.shape[0], top_left[1], bottom_left[1], top_right[1], bottom_right[1]]))
        offset = [abs(int(min([0, img_pre.shape[1], top_left[0], bottom_left[0], top_right[0], bottom_right[0]]))),
                    abs(int(min([0, img_pre.shape[0], top_left[1], bottom_left[1], top_right[1], bottom_right[1]])))]
        dsize = (cx + offset[0], cy + offset[1])
        print("image dsize: ", dsize, "offset", offset)

        top_left[0:2] += offset
        bottom_left[0:2] += offset
        top_right[0:2] += offset
        bottom_right[0:2] += offset

        dstpoints = np.array([top_left, bottom_left, top_right, bottom_right])
        srcpoints = np.array([[0, 0], [0, img_pre.shape[0]], [img_pre.shape[1], 0], [img_pre.shape[1], img_pre.shape[0]]])

        H_offset, _ = cv2.findHomography(srcpoints, dstpoints)

        warped_img2 = cv2.warpPerspective(img_pre, H_offset, dsize)
        # cv2.imwrite("warped1.jpg", warped_img2)

        warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
        warped_img1[offset[1]:img_append.shape[0] + offset[1], offset[0]:img_append.shape[1] + offset[0]] = img_append
        tmp = blend_linear(warped_img1, warped_img2)
        img_pre = tmp

    left_image = tmp

    # right_list = list(reversed(right_list))
    for img_append in right_list:
        H = match_H(left_image, img_append)
        print(H)

        bottom_right = np.dot(H, np.array([img_append.shape[1], img_append.shape[0], 1]))
        top_left = np.dot(H, np.array([0, 0, 1]))
        bottom_left = np.dot(H, np.array([0, img_append.shape[0], 1]))
        top_right = np.dot(H, np.array([img_append.shape[1], 0, 1]))

        bottom_right /= bottom_right[-1]
        top_left /= top_left[-1]
        bottom_left /= bottom_left[-1]
        top_right /= top_right[-1]

        cx = int(max([0, left_image.shape[1], top_left[0], bottom_left[0], top_right[0], bottom_right[0]]))
        cy = int(max([0, left_image.shape[0], top_left[1], bottom_left[1], top_right[1], bottom_right[1]]))
        offset = [abs(int(min([0, left_image.shape[1], top_left[0], bottom_left[0], top_right[0], bottom_right[0]]))),
                    abs(int(min([0, left_image.shape[0], top_left[1], bottom_left[1], top_right[1], bottom_right[1]])))]
        dsize = (cx + offset[0], cy + offset[1])
        print("image dsize: ", dsize, "offset", offset)

        top_left[0:2] += offset
        bottom_left[0:2] += offset
        top_right[0:2] += offset
        bottom_right[0:2] += offset

        dstpoints = np.array([top_left, bottom_left, top_right, bottom_right])
        srcpoints = np.array([[0, 0], [0, img_append.shape[0]], [img_append.shape[1], 0], [img_append.shape[1], img_append.shape[0]]])
        
        H_offset, _ = cv2.findHomography(dstpoints, srcpoints)
        
        warped_img2 = cv2.warpPerspective(img_append, H_offset, dsize, flags=cv2.WARP_INVERSE_MAP)
        # cv2.imwrite("warped2.jpg", warped_img2)
        warped_img1 = np.zeros([dsize[1], dsize[0], 3], np.uint8)
        warped_img1[offset[1]:left_image.shape[0] + offset[1], offset[0]:left_image.shape[1] + offset[0]] = left_image
        tmp = blend_linear(warped_img1, warped_img2)
        left_image = tmp

    return left_image

def blend_linear(img1, img2):
    mask1 = ((img1[:,:,0] | img1[:,:,1] | img1[:,:,2]) > 0)
    mask2 = ((img2[:,:,0] | img2[:,:,1] | img2[:,:,2]) > 0)

    r,c = np.nonzero(mask1)
    center1 = [np.mean(r), np.mean(c)]

    r,c = np.nonzero(mask2)
    center2 = [np.mean(r), np.mean(c)]

    vec = np.array(center2) - np.array(center1)
    intsct_mask = mask1 & mask2

    r,c = np.nonzero(intsct_mask)

    out_wmask = np.zeros(mask2.shape[:2])
    proj_val = (r - center1[0]) * vec[0] + (c - center1[1]) * vec[1]
    # FIXME: proj_val is empty

    out_wmask[r,c] = (proj_val - (min(proj_val)+(1e-3))) / ((max(proj_val)-(1e-3)) - (min(proj_val)+(1e-3)))

    # blending
    mask1 = mask1 & (out_wmask == 0)
    mask2 = out_wmask
    mask3 = mask2 & (out_wmask == 0)

    out = np.zeros(img1.shape)
    for i in range(3):
        out[:,:,i] = img1[:,:,i] * (mask1 + (1 - mask2) * (mask2 != 0)) + img2[:,:,i] * (mask2 + mask3)
    
    return np.uint8(out)

def match_H(img1, img2, mode='sift', k=2, r=0.7, threshold=5.0, max_iters=2000, is_default=True, is_show=False):
    kp1, kp2, matches = match(img1, img2, mode=mode, k=k, r=r, is_default=True)
    H, _ = solve_homography(kp1, kp2, matches, threshold=threshold, max_iters=max_iters, is_default=True)

    return H

def main(dir, out):
    left_list, right_list = read_img_list(dir)
    res = left_shift(left_list, right_list)

    cv2.imwrite(out, res)


if __name__ == '__main__':
    main('../data/data1/', '../res/data1.jpg')
    main('../data/data2/', '../res/data2.jpg')
    main('../data/data3/', '../res/data3.jpg')
    main('../data/data4/', '../res/data4.jpg')
