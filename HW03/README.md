# Scene Recognition with Bag of Words

## Environment

*  Python 3.7+ required

```bash
cd code
pip install -r requirements.txt
```



## Tiny Images Representation

### Running

```bash
cd code
python tiny_images_representation.py
```

### Implementation  
1. Tiny Image Feature 

   Simply follow the instruction and scale the image size to 16x16. Moreover, a normalization is applied to each tiny image.  Checkout function `get_tiny_images()` in `tiny_iamges_representation.py` 

2. KNN
   
   To calculate L2 distance, I wrote a function `euclidean_distances(x,y)` to compute L2 distance between each row vector in matrix x and y.  
   
    ```python
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
    ```
   After that, the operation is simple. Checkout function `knn_classifier()` in `utils.py`. Notice that the parameter k is changeable. The default value is 1.

### Results
```
Average accuracy: 0.24
Bedroom: 0.15
Coast: 0.37
Forest: 0.1
Highway: 0.6
Industrial: 0.1
InsideCity: 0.07
Kitchen: 0.16
LivingRoom: 0.09
Mountain: 0.31
Office: 0.17
OpenCountry: 0.41
Store: 0.04
Street: 0.48
Suburb: 0.39
TallBuilding: 0.16
```



## Bag of SIFT Representation
### Running
```bash
cd code
python bag_of_sift_representation.py
```
### Implementation

1. Building the vocabulary of visual words

   To extract SIFT feature, I used `cv2.SIFT.create()` from OpenCV and wrote the function `calcSIFT()`.

   ```python
   def calcSIFT(img, stride=10, size=16):
       sift = cv2.SIFT_create()
       kp = [cv2.KeyPoint(x, y, stride) for y in range(int(size/2), img.shape[0], stride) 
                                       for x in range(int(size/2), img.shape[1], stride)]
       _, des = sift.compute(img, kp)
       return des
   ```

   To create the library, we are going to sample the feature based on SIFT descriptor and then clustering them to get the category centers.  I used `cv2.kmeans()` from OpenCV. Notice that the parameters can be changed.

   ```python
   criteria = (cv2.TERM_CRITERIA_EPS +
               cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1)
   compactness, labels, centers = cv2.kmeans(bag_of_features, 
   										vocab_size=200, 
                                           bestLabels=None, 
                                           criteria=criteria, 
                                           attempts=20, 
                                           flags=cv2.KMEANS_PP_CENTERS)
   vocab = np.vstack(centers)
   ```

   Basically, the larger the vocabulary size, the better the prediction results.

2. Represent images as histograms of visual words

   We use L2 distance to measure which cluster the descriptor belongs, creating corresponding histograms of visual words of each image. A normalization is adapted to the histogram.

3. KNN

   That's clear enough.

### Results

```
Average accuracy: 0.38466666666666666
Bedroom: 0.17
Coast: 0.4
Forest: 0.78
Highway: 0.61
Industrial: 0.16
InsideCity: 0.31
Kitchen: 0.31
LivingRoom: 0.31
Mountain: 0.35
Office: 0.46
OpenCountry: 0.29
Store: 0.33
Street: 0.39
Suburb: 0.53
TallBuilding: 0.37
```



## Analysis and Conclusion

### Comparison

|             Method              | Accuracy |                      Confusion Matrix                      |
| :-----------------------------: | :------: | :--------------------------------------------------------: |
| Tiny Image Representation + KNN |   24%    | <img src="results\tiny_image_knn.png" style="zoom:72%;" /> |
|       Bag of SIFTs + KNN        |  38.5%   |    <img src="results\bow_knn.png" style="zoom:72%;" />     |

### Some example results 

|   Category   |                     Ground Truth Example                     |                    Correct Result Example                    |                   Incorrect Result Example                   |
| :----------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   Bedroom    | ![image_0001](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0001.jpg) | ![image_0047](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0047.jpg) | ![image_0035](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0035.jpg) |
|    Coast     | ![image_0006](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0006.jpg) | ![image_0012](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0012.jpg) | ![image_0007](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0007.jpg) |
|    Forest    | ![image_0003](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0003.jpg) | ![image_0212](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0212.jpg) | ![image_0142](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0142.jpg) |
|   Highway    | ![image_0009](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0009.jpg) | ![image_0013](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0013.jpg) | ![image_0011](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0011.jpg) |
|  Industrial  | ![image_0004](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0004.jpg) | ![image_0122](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0122.jpg) | ![image_0355](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0355.jpg) |
|  InsideCity  | ![image_0032](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0032.jpg) | ![image_0091](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0091.jpg) | ![image_0283](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0283.jpg) |
|   Kitchen    | ![image_0079](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0079.jpg) | ![image_0147](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0147.jpg) | ![image_0168](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0168.jpg) |
|  LivingRoom  | ![image_0286](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0286.jpg) | ![image_0284](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0284.jpg) | ![image_0204](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0204.jpg) |
|   Mountain   | ![image_0024](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0024.jpg) | ![image_0090](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0090.jpg) | ![image_0231](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0231.jpg) |
|    Office    | ![image_0036](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0036.jpg) | ![image_0147](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0147.jpg) | ![image_0136](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0136.jpg) |
| OpenCountry  | ![image_0070](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0070.jpg) | ![image_0114](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0114.jpg) | ![image_0374](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0374.jpg) |
|    Store     | ![image_0005](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0005.jpg) | ![image_0111](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0111.jpg) | ![image_0254](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0254.jpg) |
|    Street    | ![image_0019](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0019.jpg) | ![image_0230](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0230.jpg) | ![image_0025](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0025.jpg) |
|    Suburb    | ![image_0058](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0058.jpg) | ![image_0225](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0225.jpg) | ![image_0161](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0161.jpg) |
| TallBuilding | ![image_0041](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0041.jpg) | ![image_0031](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0031.jpg) | ![image_0332](https://ruin-typora.oss-cn-beijing.aliyuncs.com/image_0332.jpg) |

