A point cloud classification on real data using CNNs. 
There are three methods proposed in this work: 3DmFFV-Inception, 3DmFV-Net and dynamic graph convolution (DGCNN).
* Dependencies
   * tensorflow
   * open3d
   * scikit-learn
   * scipy
* Usage
    * ToDo
* ToDo
    * How to train
* Evaluation
  * The evaluation has been done on three different datasets
    * Washington RGB-D
    * JHUIT-50
    * Our industrial object dataset (b-it-bots@Work)

* Training on cluster
  * Environment setup
    * [Install anaconda](https://docs.anaconda.com/anaconda/install/linux/)  
    * Create conda environment
      ```
      conda env create -f environment.yml
      ```
  * Job script on batch system
    ```
    #!/bin/bash
    #SBATCH --job-name=train-3dmfv-net-tf
    #SBATCH --partition=gpu
    #SBATCH --nodes=1                # number of nodes
    #SBATCH --mem=64GB               # memory per node in MB (different units with$
    #SBATCH --ntasks-per-node=16    # number of cores
    #SBATCH --time=24:00:00           # HH-MM-SS
    #SBATCH --output log/job_tf.%N.%j.out # filename for STDOUT (%N: nodename, %j: j$
    #SBATCH --error log/job_tf.%N.%j.err  # filename for STDERR

    # load cuda
    module load cuda

    # activate environment
    source ~/anaconda3/bin/activate ~/anaconda3/envs/env-pointcloud

    # locate to your root directory 
    cd /home/mwasil2s/deeplearning_walkthrough/3DmFV-Net

    # run the script
    python train_fvnet_inception.py --dataset_name=rgbd_washington --num_gaussians=12 --fvnet=0 --num_inception=5 --normalize=1
    ```