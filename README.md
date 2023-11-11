 # RPFeat : Robust Planetary Feature
 This repository contains the implementation of the following paper:
 ```
Zhong, J., Yan, J., Li, M., & Barriot, J. P. (2023). A deep learning-based local feature extraction method for improved image matching and surface reconstruction from Yutu-2 PCAM images on the Moon. ISPRS Journal of Photogrammetry and Remote Sensing, 206, 16-29.
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

## Acknowledgment
We acknowledge the contributions of the following open-source projects and their authors：
```
https://github.com/colmap/colmap
https://github.com/cavalli1234/AdaLAM
https://github.com/naver/r2d2
https://github.com/Xbbei/super-colmap
```

## Citation
If you find our research useful, please cite this paper:
```
@article{zhong2023deep,
  title={A deep learning-based local feature extraction method for improved image matching and surface reconstruction from Yutu-2 PCAM images on the Moon},
  author={Zhong, Jiageng and Yan, Jianguo and Li, Ming and Barriot, Jean-Pierre},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={206},
  pages={16--29},
  year={2023},
  publisher={Elsevier}
}
```
