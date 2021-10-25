#!/bin/bash --login

########## SBATCH Lines for Resource Request ##########
#SBATCH --time=5:00:00
#SBATCH -N 1 -n 1 -c 2
#SBATCH --mem-per-cpu=64G            # memory required per allocated CPU (or core) - amount of memory (in bytes)        
#SBATCH --job-name=test
#SBATCH --mail-user=lixiaoy5@msu.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name scRNA_seq_inte


########## Command Lines to Run ##########
# module swap GNU Intel
module purge
module load GCCcore/10.3.0

# cd ${SLURM_SUBMIT_DIR}       ### change to the directory where your code is located

### call your executable (similar to mpirun)
conda activate env_
python -u node_feature.py > output.txt


scontrol show job $SLURM_JOB_ID           ### write job information to output file
js -j $SLURM_JOB_ID 