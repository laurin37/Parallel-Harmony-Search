#!/bin/bash
#PBS -l select=1:ncpus=64:mem=2gb
# set max execution time
#PBS -l walltime=6:00:00
# imposta la coda di esecuzione
#PBS -q short_cpuQ

module load mpich-3.2

#Serial
#./Parallel-Harmony-Search/src/serial.out 1000 0.9 0.3 0.01 1000 5 42
#./Parallel-Harmony-Search/src/serial.out 1000 0.9 0.3 0.01 10000 5 42
#./Parallel-Harmony-Search/src/serial.out 1000 0.9 0.3 0.01 100000 5 42
#./Parallel-Harmony-Search/src/serial.out 1000 0.9 0.3 0.01 1000000 5 42
#./Parallel-Harmony-Search/src/serial.out 1000 0.9 0.3 0.01 10000000 5 42

#Parallel MPI
./Parallel-Harmony-Search/src/serial.out 1000 0.9 0.3 0.01 10000000 36 42
mpirun.actual -np 1 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 10000000 36 42
mpirun.actual -np 2 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 20000000 36 42
mpirun.actual -np 4 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 40000000 36 42
mpirun.actual -np 8 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 80000000 36 42
mpirun.actual -np 16 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 160000000 36 42
mpirun.actual -np 32 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 320000000 36 42
mpirun.actual -np 64 ./Parallel-Harmony-Search/src/mpi.out 1000 0.9 0.3 0.01 640000000 36 42


#Parallel OpenMP
#./Parallel-Harmony-Search/src/openmp.out 1000 0.9 0.3 0.01 100000000 30 42 4
#./Parallel-Harmony-Search/src/openmp.out 1000 0.9 0.3 0.01 100000000 30 42 8
#./Parallel-Harmony-Search/src/openmp.out 1000 0.9 0.3 0.01 100000000 30 42 16
#./Parallel-Harmony-Search/src/openmp.out 1000 0.9 0.3 0.01 100000000 30 42 32
#./Parallel-Harmony-Search/src/openmp.out 1000 0.9 0.3 0.01 100000000 30 42 64

#Parallel Hybrid