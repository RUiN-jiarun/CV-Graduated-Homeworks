# Panorama Stitching

## Environment

* Python 3.7+ required

```bash
cd code
pip install -r requirements.txt
```

## Feature Detection, Descriptor and Matching

### Running

```bash
cd code
python feature_matching.py
```

Please check `feature_matching.py` before running. Path to the images may need to be changed. 

### Results

* SIFT detector & descriptor

  ![](res/data2/match.jpg)

* SIFT detector & concatenated pixel values descriptor

  ![](/res/data2/match_cpv.jpg)

## Estimation of Homography

### Running

```bash
cd code
python homography.py
```

Please check `homography.py` before running. Path to the images may need to be changed.

### Results

* SIFT detector & descriptor + RANSAC

  ![](res/data2/ransac_res.jpg)

* SIFT detector & concatenated pixel values descriptor + RANSAC

  ![](/res/data2/ransac_res_cpv.jpg)

## Image Stitching

### Running

```bash
cd code
python panorama.py
```

Please check `panorama.py` before running. Path to the images may need to be changed.

### Results

![](res/data2.jpg)