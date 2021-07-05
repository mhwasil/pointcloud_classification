# Point Cloud Classification Using Deep Learning and Classical Machine Learning Methods

Supported models are available under `config/config.yaml`.

## Usage

* Dependencies
  ```
  conda env create -f environment.yml
  ```
* Usage
  * Train
    ```
    python trainer.py --model DGCNNC --train
    ```
  * Evaluation
    ```
    python trainer.py --model DGCNNC 
    ```

## Adding your own model

ToDo

## [Training on H-BRS cluster](https://github.com/mhwasil/pointcloud_classification/blob/master/hbrs_cluster_usage.md)

## Acknowledgement
* [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://github.com/charlesq34/pointnet)
* [PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space](https://github.com/charlesq34/pointnet2)
* [Dynamic Graph CNN for Learning on Point Clouds](https://github.com/WangYueFt/dgcnn)
* [3DmFV : Three-Dimensional Point Cloud Classification in Real-Time using Convolutional Neural Networks](https://github.com/sitzikbs/3DmFV-Net)
* [PointCNN: Convolution On X-Transformed Points](https://github.com/yangyanli/PointCNN)
* [SpiderCNN: Deep Learning on Point Sets with Parameterized Convolutional Filters](https://github.com/xyf513/SpiderCNN)