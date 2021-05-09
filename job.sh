#!/bin/bash
#SBATCH --job-name=train-pc-cls
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # number of nodes
#SBATCH --mem=64GB                      # memory per node in MB (different units with$
#SBATCH --ntasks-per-node=16            # number of cores
#SBATCH --time=24:00:00                 # HH-MM-SS
#SBATCH --output log/job_tf.%N.%j.out   # filename for STDOUT (%N: nodename, %j: j$
#SBATCH --error log/job_tf.%N.%j.err    # filename for STDERR

# load cuda
module load cuda/9.2

# activate environment
source ~/anaconda3/bin/activate ~/anaconda3/envs/pc-classification

# locate to your root directory 
cd /home/mwasil2s/deeplearning_walkthrough/pointcloud_classification

# run the script
python train_fvnet_inception.py --dataset_name=rgbd_washington --num_gaussians=12 --fvnet=0 --num_inception=5 --normalize=1