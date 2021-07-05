### Training on H-BRS Scientific Computing Cluster

[https://wr0.wr.inf.h-brs.de/](https://wr0.wr.inf.h-brs.de/)

* Request account
  * You can request access to [H-BRS cluster](https://wr0.wr.inf.h-brs.de/wr/index.html) by sending email to the responsible persons from MAS-Group.
* Environment setup
  * ssh to the cluster
    ```
    ssh username2s@wr0.wr.inf.h-brs.de
    ```
  * [Install anaconda](https://docs.anaconda.com/anaconda/install/linux/)

  * Create conda environment
    * Enter your conda
      ```
      source ~/anaconda3/bin/activate
      ```
    * Create your environment
      ```
      conda env create -f environment.yml
      ```
      check environment.yml and change the dependencies if necessary

* Create a job script on batch system
  ```
  #!/bin/bash
  #SBATCH --job-name=my_batch_job
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

  # locate to pointcloud_classification root directory 
  cd ~/pointcloud_classification

  # run the script
  python trainer.py --model DGCNNC --train
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
