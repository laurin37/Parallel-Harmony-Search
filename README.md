# Parallel-Harmony-Search
This is parallelization of Harmony Search Algorithm. 

# compile Serial
```
g++ -std=c++11 -g -Wall -o serial.out harmonySearchSerial.cpp
```

# compile MPI
```
mpic++ -std=c++11 -g -Wall harmonySearchParallelMPI.cpp -o mpi.out
```

# compile OpenMP
```
g++ -std=c++11 -o openmp.out harmonySearchParallelOpenMP.cpp
```

# compile Hybrid
```
g++ -std=c++11 -o hybrid.out harmonySearchParallelHybrid.cpp
```


# submit to hpc
```
qsub harmonySearch.sh
```