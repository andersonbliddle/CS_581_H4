#!/bin/bash
source /apps/profiles/modules_asax.sh.dyn
module load openmpi/4.1.4-gcc11

mpirun -n 1 ./hw4 5000 5000 1 /scratch/ualclsd0173/1/ 1
mpirun -n 2 ./hw4 5000 5000 2 /scratch/ualclsd0173/2/ 1
mpirun -n 4 ./hw4 5000 5000 4 /scratch/ualclsd0173/4/ 1
mpirun -n 8 ./hw4 5000 5000 8 /scratch/ualclsd0173/8/ 1
mpirun -n 10 ./hw4 5000 5000 10 /scratch/ualclsd0173/10/ 1
mpirun -n 16 ./hw4 5000 5000 16 /scratch/ualclsd0173/16/ 1
mpirun -n 20 ./hw4 5000 5000 20 /scratch/ualclsd0173/20/ 1

mpirun -n 1 ./hw4 5000 5000 1 /scratch/ualclsd0173/1/ 1
mpirun -n 2 ./hw4 5000 5000 2 /scratch/ualclsd0173/2/ 1
mpirun -n 4 ./hw4 5000 5000 4 /scratch/ualclsd0173/4/ 1
mpirun -n 8 ./hw4 5000 5000 8 /scratch/ualclsd0173/8/ 1
mpirun -n 10 ./hw4 5000 5000 10 /scratch/ualclsd0173/10/ 1
mpirun -n 16 ./hw4 5000 5000 16 /scratch/ualclsd0173/16/ 1
mpirun -n 20 ./hw4 5000 5000 20 /scratch/ualclsd0173/20/ 1

mpirun -n 1 ./hw4 5000 5000 1 /scratch/ualclsd0173/1/ 1
mpirun -n 2 ./hw4 5000 5000 2 /scratch/ualclsd0173/2/ 1
mpirun -n 4 ./hw4 5000 5000 4 /scratch/ualclsd0173/4/ 1
mpirun -n 8 ./hw4 5000 5000 8 /scratch/ualclsd0173/8/ 1
mpirun -n 10 ./hw4 5000 5000 10 /scratch/ualclsd0173/10/ 1
mpirun -n 16 ./hw4 5000 5000 16 /scratch/ualclsd0173/16/ 1
mpirun -n 20 ./hw4 5000 5000 20 /scratch/ualclsd0173/20/ 1
