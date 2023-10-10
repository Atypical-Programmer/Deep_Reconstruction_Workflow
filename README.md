 # RPFeat : Robust Planetary Feature
 This repository contains the implementation of the following paper:
 ```
 coming soon ...
 ```

 ## Configuration
 Our codes are tested on CentOS Linux release 8.5.2111, and NVIDIA graphics card is required. (We choose NVIDIA GeForce RTX 3090)
 
 We recommand to use [Anaconda](https://www.anaconda.com/) to deploy the environment. Install with conda:
 ```
 conda env create -f env.yaml
 conda activate cv
 ```

Besides, [AdaLAM](https://github.com/cavalli1234/AdaLAM) and [COLMAP](https://github.com/colmap/colmap) are also required, and you can install them according to their official tutorials.

## Preparation
First of all, you need to prepare data and weights.

The code expects folders structure as follows.
```
PROJECT_DIR/
    images/
        number1_images.png
        number2_images.png
        number3_images.png
        ...
```

The pretrained models are available [here](https://drive.google.com/drive/folders/1y6B6DD7PdhbKT--CjL3hKnZiBS3pkM-m?usp=drive_link), and you need to set the path for weights in [RPFeatDetectors.py](RPFeatDetectors.py).

## Feature extraction and matching
To extract keypoints and match them, you can run:
```
python generateFeatures.py --dir PATH_TO_PROJECT_DIR/  --match 0 # using ratio-test for matching
# or
python generateFeatures.py --dir PATH_TO_PROJECT_DIR/  --match 1 # using AdaLAM for matching
```
## Sparse Reconstruction
To perform sparse reconstruction, you can run:
```
python Reconstruction.py --dir PATH_TO_PROJECT_DIR/
```
The parameters can be modified in the file.
## Dense Reconstruction
If you want to perform dense reconstruction, you can then run:
```
python DenseReconstruction.py --dir PATH_TO_PROJECT_DIR/
```
The parameters can be modified in the file.

## Citation
If you find our research useful, please cite this paper:
```
coming soon ...
```
