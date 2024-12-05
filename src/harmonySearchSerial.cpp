#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <stdexcept>
#include <functional>
#include <map>

#define LOGGING

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

ObjectiveFunction createMichalewicz(int dimensions) 
{
    ObjectiveFunction michalewicz
    {
        "Michalewicz",
        dimensions,
        0.0,
        M_PI,
        [](const std::vector<double>& vars) 
        {
            double sum = 0.0;
            for (size_t i = 0; i < vars.size(); ++i) {
                sum += -sin(vars[i]) * std::pow(sin((i + 1) * vars[i] * vars[i] / M_PI), 20);
            }
            return sum;
        },
        {2.20319, 1.57049}, // Real solution for 2D case
        -1.801              // Real value at the solution
    };
    return michalewicz;
}

// Harmony Search Algorithm
double harmonySearch(const ObjectiveFunction& objFunc, int memorySize, double harmonyMemoryConsideringRate,
                     double pitchAdjustingRate, double bandwidth, int maxIterations, int logInterval = 1000) 
{
    auto start = std::chrono::high_resolution_clock::now();                    

    // Initialize Harmony Memory (HM)
    std::vector<std::vector<double>> harmonyMemory(memorySize);
    std::vector<double> harmonyMemoryFitness(memorySize);

    // Populate initial harmony memory
    for (int i = 0; i < memorySize; ++i) 
    {
        harmonyMemory[i] = generateRandomSolution(objFunc.dimensions, objFunc.min, objFunc.max);
        harmonyMemoryFitness[i] = objFunc.func(harmonyMemory[i]);
    }

    // Track the best solution
    std::vector<double> bestSolution = harmonyMemory[0];
    double bestFitness = harmonyMemoryFitness[0];

    // Main optimization loop
    for (int iter = 0; iter < maxIterations; ++iter) 
    {
        std::vector<double> newSolution = generateRandomSolution(objFunc.dimensions, objFunc.min, objFunc.max);

        // Harmony Memory Considering Rate (HMCR)
        if (randomDouble(0, 1) < harmonyMemoryConsideringRate) 
        {
            int randIndex = rand() % memorySize;
            newSolution = harmonyMemory[randIndex];

            // Pitch Adjusting Rate (PAR)
            if (randomDouble(0, 1) < pitchAdjustingRate) 
            {
                pitchAdjust(newSolution, bandwidth, objFunc.min, objFunc.max);
            }
        }

        // Evaluate the new solution
        double newFitness = objFunc.func(newSolution);

        // Replace the worst harmony if the new one is better
        int worstIndex = 0;
        for (int i = 1; i < memorySize; ++i) {
            if (harmonyMemoryFitness[i] > harmonyMemoryFitness[worstIndex]) 
            {
                worstIndex = i;
            }
        }

        if (newFitness < harmonyMemoryFitness[worstIndex]) 
        {
            harmonyMemory[worstIndex] = newSolution;
            harmonyMemoryFitness[worstIndex] = newFitness;
        }

        // Update the best solution found
        if (newFitness < bestFitness) 
        {
            bestSolution = newSolution;
            bestFitness = newFitness;
        }
        #ifdef LOGGING
        // Log progress
        if (iter % logInterval == 0) 
        {
            std::cout << "Iteration " << iter << " - Best fitness so far: " << bestFitness << std::endl;
        }
        #endif
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    // Print final results
    std::cout << "Objective Function: " << objFunc.name << "\n";
    std::cout << "Best solution found: ";
    for (const auto& var : bestSolution) std::cout << var << " ";
    std::cout << "\nFunction value at best solution: " << bestFitness << "\n";

    // Print real solution (if available)
    if (!objFunc.realParameters.empty()) 
    {
        std::cout << "Real solution: ";
        for (const auto& param : objFunc.realParameters) std::cout << param << " ";
        std::cout << "\nReal function value: " << objFunc.realValue << "\n";
    }
    return bestFitness;
}

int main(int argc, char *argv[]) 
{
    // Default values for Harmony Search parameters
    int memorySize = 1000;
    int maxIterations = 100000;
    double harmonyMemoryConsideringRate = 0.9;
    double pitchAdjustingRate = 0.3;
    double bandwidth = 0.01;
    int seed = rd();

    // Read parameters from command line if provided
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
        std::cerr << "Invalid argument: please enter numeric values for all parameters." << std::endl;
        return 1;
    }

    // Set seed for reproducibility
    gen.seed(seed);

    // Print parameter values
    std::cout << "memory Size = " << memorySize << std::endl;
    std::cout << "max Iterations = " << maxIterations << std::endl;
    std::cout << "harmony Memory Considering Rate = " << harmonyMemoryConsideringRate << std::endl;
    std::cout << "pitch Adjusting Rate = " << pitchAdjustingRate << std::endl;
    std::cout << "band width = " << bandwidth << std::endl;
    std::cout << "seed = " << seed << std::endl;

   // Create a vector of objective functions
    std::vector<ObjectiveFunction> objectiveFunctions = 
    {
        createRosenbrock(10),  // 10D Rosenbrock
        //createMichalewicz(2) // 2D Michalewicz
    };

    // Iterate through each objective function and run Harmony Search
    for (const auto& objectiveFunction : objectiveFunctions) 
    {
        std::cout << "\nOptimizing " << objectiveFunction.name << " Function\n";
        harmonySearch(objectiveFunction, memorySize, harmonyMemoryConsideringRate, pitchAdjustingRate, bandwidth, maxIterations, int(maxIterations / 10));
    }

    return 0;
}