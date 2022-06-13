#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.%j.%N.out"
#SBATCH --partition=CLUSTER
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
# #SBATCH --mem=512M
# #SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:08:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
# module purge
# module load cpu
#Load module file(s) into the shell environment
# module load gcc
# module load rocks-openmpi
# module load slurm
srun --mpi=openmpi -n 2 ./apf-ref -n 800 -i 2000 -x 2 -y 1
mpirun -n 2 ./apf-ref -n 800 -i 2000 -x 2 -y 1