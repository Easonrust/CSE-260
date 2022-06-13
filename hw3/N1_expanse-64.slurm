#!/bin/bash
#SBATCH --job-name="./apf"
#SBATCH --output="apf.64.1800.100000.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --mem=64G
#SBATCH --account=csd720
# #SBATCH --export=None
#SBATCH --export=ALL
#SBATCH -t 0:03:00
####   #SBATCH --mail-type=BEGIN,END,FAIL
####   #SBATCH --mail-user=your_email@ucsd.edu

# setup your environment

export SLURM_EXPORT_ENV=ALL
module purge
module load cpu
#Load module file(s) into the shell environment
module load gcc/9.2.0
module load mvapich2/2.3.6
module load slurm
srun --mpi=pmi2 -n 64 ./apf -n 1800 -i 100000 -x 4 -y 16