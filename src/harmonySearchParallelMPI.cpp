#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <stdexcept>
#include <functional>
#include <mpi.h>

// Random Number Generator setup
std::random_device rd;
std::mt19937 gen(rd());

// Clamps a value within the range [min, max]
double clamp(double value, double min, double max) 
{
    return (value < min) ? min : (value > max ? max : value);
}

// Generates a random double within the specified range [min, max]
double randomDouble(double min, double max) 
{
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Generates a random solution vector of specified dimensions within [min, max] for each variable
std::vector<double> generateRandomSolution(int dimensions, double min, double max) 
{
    std::vector<double> solution(dimensions);
    for (int i = 0; i < dimensions; ++i) 
    {
        solution[i] = randomDouble(min, max);
    }
    return solution;
}

// Adjusts the pitch of a solution vector within a specified bandwidth
void pitchAdjust(std::vector<double>& solution, double bandwidth, double min, double max) 
{
    for (double& value : solution) 
    {
        value += randomDouble(-bandwidth, bandwidth);
        value = clamp(value, min, max);
    }
}

// Objective Function Interface
class ObjectiveFunction 
{
    public:
        std::string name;                             // Name of the function
        int dimensions;                               // Number of parameters (variables)
        double min;                                   // Minimum bound for parameters
        double max;                                   // Maximum bound for parameters
        std::function<double(const std::vector<double>&)> func; // Function definition
        std::vector<double> realParameters;           // Real solution (for comparison)
        double realValue;                             // Real value of the function (for comparison)
};

// Example Objective Functions
ObjectiveFunction createRosenbrock(int dimensions) 
{
    ObjectiveFunction rosenbrock
    {
        "Rosenbrock",
        dimensions,
        -5.0,
        10.0,
        [](const std::vector<double>& vars) 
        {
            double sum = 0.0;
            for (size_t i = 0; i < vars.size() - 1; ++i) 
            {
                sum += 100 * std::pow(vars[i + 1] - vars[i] * vars[i], 2) + std::pow(1 - vars[i], 2);
            }
            return sum;
        },
        std::vector<double>(dimensions, 1.0), // Real solution is (1, 1, ..., 1)
        0.0                                   // Real value at the solution is 0
    };
    return rosenbrock;
}

// Harmony Search Algorithm
double harmonySearch(const ObjectiveFunction& objFunc, int memorySize, double harmonyMemoryConsideringRate,
                     double pitchAdjustingRate, double bandwidth, int maxIterations, int logInterval, 
                     int rank, int size) 
{
    auto start = std::chrono::high_resolution_clock::now();                    

    // Determine chunk size for each MPI process
    int chunkSize = memorySize / size;
    //int startIdx = rank * chunkSize;
    //int endIdx = (rank == size - 1) ? memorySize : startIdx + chunkSize;

    // Local memory for each process
    std::vector<std::vector<double>> localMemory(chunkSize);
    std::vector<double> localFitness(chunkSize);

    // Populate local harmony memory
    for (int i = 0; i < chunkSize; ++i) 
    {
        localMemory[i] = generateRandomSolution(objFunc.dimensions, objFunc.min, objFunc.max);
        localFitness[i] = objFunc.func(localMemory[i]);
    }

    // Track the best solution locally
    double localBestFitness = localFitness[0];
    std::vector<double> localBestSolution = localMemory[0];
    for (int i = 1; i < chunkSize; ++i) 
    {
        if (localFitness[i] < localBestFitness) 
        {
            localBestFitness = localFitness[i];
            localBestSolution = localMemory[i];
        }
    }

    // Main optimization loop
    for (int iter = 0; iter < maxIterations; ++iter) 
    {
        std::vector<double> newSolution = generateRandomSolution(objFunc.dimensions, objFunc.min, objFunc.max);

        // Harmony Memory Considering Rate (HMCR)
        if (randomDouble(0, 1) < harmonyMemoryConsideringRate) 
        {
            int randIndex = rand() % chunkSize;
            newSolution = localMemory[randIndex];

            // Pitch Adjusting Rate (PAR)
            if (randomDouble(0, 1) < pitchAdjustingRate) 
            {
                pitchAdjust(newSolution, bandwidth, objFunc.min, objFunc.max);
            }
        }

        // Evaluate the new solution
        double newFitness = objFunc.func(newSolution);

        // Replace the worst harmony locally if the new one is better
        int worstIndex = 0;
        for (int i = 1; i < chunkSize; ++i) {
            if (localFitness[i] > localFitness[worstIndex]) 
            {
                worstIndex = i;
            }
        }

        if (newFitness < localFitness[worstIndex]) 
        {
            localMemory[worstIndex] = newSolution;
            localFitness[worstIndex] = newFitness;
        }

        // Update the best solution found locally
        if (newFitness < localBestFitness) 
        {
            localBestFitness = newFitness;
            localBestSolution = newSolution;
        }

        // Synchronize global best solution across processes every few iterations
        if (iter % logInterval == 0) 
        {
            double globalBestFitness;
            MPI_Allreduce(&localBestFitness, &globalBestFitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

            // Log progress (rank 0 only)
            if (rank == 0) 
            {
                std::cout << "Iteration " << iter << " - Best fitness so far: " << globalBestFitness << std::endl;
            }
        }
    }

    // Gather best solution from all processes
    double globalBestFitness;
    MPI_Allreduce(&localBestFitness, &globalBestFitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);

    // Broadcast the global best solution
    std::vector<double> globalBestSolution = localBestSolution;
    MPI_Bcast(globalBestSolution.data(), objFunc.dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Print results (rank 0 only)
    if (rank == 0) 
    {
        std::cout << "Final Best Fitness: " << globalBestFitness << "\nBest Solution: ";
        for (const auto& val : globalBestSolution) std::cout << val << " ";
        std::cout << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    if (rank == 0) 
    {
        std::cout << "Execution time: " << duration.count() << " seconds\n";
    }

    return globalBestFitness;
}

int main(int argc, char* argv[]) 
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Harmony Search Parameters
    int memorySize = 1000, maxIterations = 100000;
    double harmonyMemoryConsideringRate = 0.9, pitchAdjustingRate = 0.3, bandwidth = 0.01;
    int seed = rd();

    // Read parameters from command line
    try 
    {
        if (argc > 1) memorySize = std::stoi(argv[1]);
        if (argc > 2) maxIterations = std::stoi(argv[2]);
        if (argc > 3) harmonyMemoryConsideringRate = std::stod(argv[3]);
        if (argc > 4) pitchAdjustingRate = std::stod(argv[4]);
        if (argc > 5) bandwidth = std::stod(argv[5]);
        if (argc > 6) seed = std::stoi(argv[6]);
    } catch (const std::invalid_argument& e) 
    {
        if (rank == 0) 
        {
            std::cerr << "Invalid argument: ensure all parameters are numeric." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Set seed for reproducibility
    gen.seed(seed);

    // Define the objective function
    auto rosenbrock = createRosenbrock(5);

    // Run Harmony Search
    harmonySearch(rosenbrock, memorySize, harmonyMemoryConsideringRate, pitchAdjustingRate, bandwidth, maxIterations, int(maxIterations / 10), rank, size);

    MPI_Finalize();
    return 0;
}
