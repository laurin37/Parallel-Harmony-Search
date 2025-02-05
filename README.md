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
g++ -g -Wall -std=c++11 -fopenmp -o openmp.out harmonySearchParallelOpenMP.cpp
```

# compile Hybrid
```
mpic++ -std=c++11 -g -Wall harmonySearchParallelHybrid.cpp -o hybrid.out
```


# submit to hpc
```
qsub harmonySearch.sh
```

# analyze run mpi
```
python analyze_hs.py
```

# analyze run omp
```
python analyze_hs_omp.py
```