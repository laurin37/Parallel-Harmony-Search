#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb
# set max execution time
#PBS -l walltime=3:00:00
# imposta la coda di esecuzione
#PBS -q short_cpuQ

module load mpich-3.2

#Serial
#./Parallel-Harmony-Search/src/serial.out 1000 100000 0.9 0.3 0.01 1
#./Parallel-Harmony-Search/src/serial.out 1000 1000000 0.9 0.3 0.01 1
#./Parallel-Harmony-Search/src/serial.out 1000 10000000 0.9 0.3 0.01 1 # 1.5 Minutes
#./Parallel-Harmony-Search/src/serial.out 1000 100000000 0.9 0.3 0.01 1 # 14 Minutes
#./Parallel-Harmony-Search/src/serial.out 1000 1000000000 0.9 0.3 0.01 1 # 2 Hours 20 Minutes

#Parallel MPI
./Parallel-Harmony-Search/src/serial.out 10000 100000 0.9 0.3 0.01 1

mpirun.actual -np 2 ./Parallel-Harmony-Search/src/mpi.out 10000 1000000 0.9 0.3 0.01 1
mpirun.actual -np 4 ./Parallel-Harmony-Search/src/mpi.out 10000 1000000 0.9 0.3 0.01 1
mpirun.actual -np 8 ./Parallel-Harmony-Search/src/mpi.out 10000 1000000 0.9 0.3 0.01 1


#Parallel OpenMP

#Parallel Hybrid