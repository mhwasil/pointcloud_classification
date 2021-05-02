A point cloud classification on real data using CNNs. 
We proposed 3DmFV-Inception and implemented 3DmFV-Net and dynamic graph convolution (DGCNN) on real dataset.
* Dependencies
   * tensorflow
   * open3d
   * scikit-learn
   * scipy
* Usage
    * ToDo
* ToDo
    * How to train
    * Upload datasets
* Evaluation
  * The evaluation has been done on three different datasets
    * Our industrial object dataset (b-it-bots@Work)

* Training on cluster
  * Request account
    * You can request access to [H-BRS cluster](https://wr0.wr.inf.h-brs.de/wr/index.html) by sending email to Deebul or Santosh
  * Environment setup
    * ssh to the cluster
      ```
      ssh username2s@wr0.wr.inf.h-brs.de
      ```
    * [Install anaconda](https://docs.anaconda.com/anaconda/install/linux/)
    * Setup anaconda shortcut
      * Create .condarc config
        ```
        #!/bin/bash
        function condapy() {
            local RED="\[\033[0;31m\]"
            PATH=$HOME/anaconda3/bin:$PATH
            export PS1="$RED[CONDA3] $PS1"
        }
        condapy
        alias ipy='jupyter qtconsole'
        alias startnb='jupyter notebook'
        alias nb2pdf='jupyter nbconvert --to latex --post PDF'
        condaforge() { conda install -c conda-forge "$@" ;}
        ```
      * Update .bashrc config
        * Add alias for condarc to bashrc
          ```
          alias anaconda3='bash --rcfile ~/.condarc'
          ```
        * The source your terminal
          ```
          source ~/.bashrc
          ```
    * Create conda environment
      * Enter your conda
        ```
        anaconda3
        ```
      * Create your environment
        ```
        conda env create -f environment.yml
        ```
        check environment.yml and change the dependencies if necessary
  * Create a job script on batch system
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
    cd /home/mwasil2s/deeplearning_walkthrough/3DmFV-Inception

    # run the script
    python train_fvnet_inception.py --dataset_name=rgbd_washington --num_gaussians=12 --fvnet=0 --num_inception=5 --normalize=1
    ```
    [More information about hardware and the current system status](https://wr0.wr.inf.h-brs.de/wr/index.html)
  * Submit the job
    ```
    sbatch job.sh
    ```
  * Status
    * Check queue
      ```
      squeue
      ```
    * Cancel the job (scancel job_id e.g. scancel 207550 for the job id 207550)
      ```
      scancel 207550
      ```
    * [More on slurm and batch systems](https://wr0.wr.inf.h-brs.de/wr/usage.html)
