# SCONet

This repository contains the code for SCONet (Segmentation Convolutional Occupancy Network) proposed in the paper:
```bibtex
@inproceedings{Jouvencel2025,
    author = {Jouvencel, Maylis and Kéchichian, Razmig and Digne, Julie and Valette, Sébastien},
    title = {SCONet: Convolutional Occupancy Networks for Multi-Organ Segmentation},
    booktitle = {IEEE ISBI},
    year = {2025}
}
```


## Installation

The necessary Python packages to generate point-clouds, train and evaluate the models can be installed by running:

```
bash env_setup.sh
```

The code is tested with PyTorch 2.5.0 and CUDA 12.4.

To generate the point-clouds, you also have to compile the code from the [3D implementation of SURF](https://github.com/valette/vtkOpenSURF3D).


## Usage

### Demo

We provide some demo point clouds if you want to test SCONet. This data includes the pre-sampled point cloud corresponding to the patients in `data/test.lst` . You can download them [here](https://kaggle.com/datasets/c6714f54d42df23b74bee21bc42a9ea314ec2b7c36e75268c858a8d5371e3557).

The pre-trained weights are stored in the folder `out/AbdomenCT-1K_demo`

To generate the segmentation maps with SCONet, you can run:

```
python generate.py configs/pointcloud/AbdomenCT-1K_demo.yaml
```


### Train from scratch

#### Data preparation

You can download the AbdomenCT-1K from the [official dataset repository](https://github.com/JunMa11/AbdomenCT-1K).

The data should be organized like this:

```
AbdomenCT-1K/
  volumes/
    id.lst
    Case_00001_0000.nii.gz
    Case_00002_0000.nii.gz
    ...
  segmentations/
    id_seg.lst
    Case_00001.nii.gz
    Case_00002.nii.gz
    ...
  train.lst
  val.lst
  test.lst
``` 

The preprocessing includes:
- the extraction of the information of the dataset in an info_dataset.csv
- the contour extraction with Canny algorithm
- the generation of the pointclouds

You can run it with the command:

```
bash preprocess.sh path/vtkOpenSURF3D/surf3d configs/pointcloud/AbdomenCT-1K.yaml
```

#### Training
To train a network from scratch, you can run the command:
```
python train.py configs/pointcloud/AbdomenCT-1K.yaml
```

#### Inference
To run an inference, you can run the command:
```
python generate.py configs/pointcloud/AbdomenCT-1K.yaml
```

#### Evaluate
To run the evaluation of the network, you can run the command:
```
python eval_seg.py configs/pointcloud/AbdomenCT-1K.yaml
```
