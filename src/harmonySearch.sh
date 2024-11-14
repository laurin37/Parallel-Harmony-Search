#!/bin/bash
#PBS -l select=1:ncpus=4:mem=2gb
# set max execution time
#PBS -l walltime=1:00:00
# imposta la coda di esecuzione
#PBS -q short_cpuQ

module load mpich-3.2

#./Parallel-Harmony-Search/src/a.out 100 1000
#./Parallel-Harmony-Search/src/a.out 100000
#./Parallel-Harmony-Search/src/a.out 10000 10000000
./Parallel-Harmony-Search/src/a.out