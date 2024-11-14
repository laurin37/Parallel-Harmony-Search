#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <string>
#include <stdexcept>
#include <functional>

// Random Number Generator setup
std::random_device rd;
std::mt19937 gen(rd());

// Generalized Objective Function: Accepts a vector of inputs
// Example objective functions: modify or add as needed
double rosenbrock(const std::vector<double>& vars) {
    double sum = 0.0;
    for (size_t i = 0; i < vars.size() - 1; ++i) {
        sum += 100 * std::pow(vars[i + 1] - vars[i] * vars[i], 2) + std::pow(1 - vars[i], 2);
    }
    return sum;
}

double michalewicz(const std::vector<double>& vars) {
    double sum = 0.0;
    for (size_t i = 0; i < vars.size(); ++i) {
        sum += -sin(vars[i]) * std::pow(sin((i + 1) * vars[i] * vars[i] / M_PI), 20);
    }
    return sum;
}

// Generates a random double within the specified range [min, max]
double randomDouble(double min, double max) {
    std::uniform_real_distribution<> dis(min, max);
    return dis(gen);
}

// Clamps a value within the range [min, max]
double clamp(double value, double min, double max) {
    return (value < min) ? min : (value > max ? max : value);
}

// Adjusts pitch of a given value within a specified bandwidth and clamps it within [0, π]
double pitchAdjust(double value, double bandwidth) {
    value += randomDouble(-bandwidth, bandwidth);
    return clamp(value, 0.0, M_PI);
}

// Generates a new random solution vector within [0, π] for each variable
std::vector<double> generateRandomSolution(int dimensions) {
    std::vector<double> solution(dimensions);
    for (int i = 0; i < dimensions; ++i) {
        solution[i] = randomDouble(0, M_PI);
    }
    return solution;
}

// Harmony Search Algorithm (generalized for any number of variables)
double harmonySearch(int memorySize, double harmonyMemoryConsideringRate, double pitchAdjustingRate, 
                     double bandwidth, int maxIterations, int dimensions, 
                     const std::function<double(const std::vector<double>&)>& objectiveFunction, 
                     int logInterval = 1000) {
    // Initialize Harmony Memory (HM)
    std::vector<std::vector<double>> harmonyMemory(memorySize); // stores solution vectors
    std::vector<double> harmonyMemoryFitness(memorySize); // stores the fitness of each solution

    // Populate initial harmony memory with random solutions within the range [0, π]
    for (int i = 0; i < memorySize; ++i) {
        harmonyMemory[i] = generateRandomSolution(dimensions);
        harmonyMemoryFitness[i] = objectiveFunction(harmonyMemory[i]);
    }

    // Initialize best solution with the first solution in harmony memory
    std::vector<double> bestSolution = harmonyMemory[0];
    double bestFitness = harmonyMemoryFitness[0];

    // Main optimization loop
    for (int iter = 0; iter < maxIterations; ++iter) {
        std::vector<double> newSolution(dimensions);

        // Harmony Memory Considering Rate (HMCR)
        if (randomDouble(0, 1) < harmonyMemoryConsideringRate) {
            int randIndex = rand() % memorySize; // Select a random solution from harmony memory
            newSolution = harmonyMemory[randIndex];

            // Pitch Adjusting Rate (PAR)
            for (int i = 0; i < dimensions; ++i) {
                if (randomDouble(0, 1) < pitchAdjustingRate) {
                    newSolution[i] = pitchAdjust(newSolution[i], bandwidth);
                }
            }
        } else {
            // Randomization: Generate a new completely random solution within [0, π]
            newSolution = generateRandomSolution(dimensions);
        }

        // Evaluate the new solution
        double newFitness = objectiveFunction(newSolution);

        // Replace the worst harmony if the new one is better
        int worstIndex = 0;
        for (int i = 1; i < memorySize; ++i) {
            if (harmonyMemoryFitness[i] > harmonyMemoryFitness[worstIndex]) {
                worstIndex = i;
            }
        }
        
        if (newFitness < harmonyMemoryFitness[worstIndex]) {
            harmonyMemory[worstIndex] = newSolution;
            harmonyMemoryFitness[worstIndex] = newFitness;
        }

        // Update the best solution found
        if (newFitness < bestFitness) {
            bestSolution = newSolution;
            bestFitness = newFitness;
        }

        // Log the best fitness at specified intervals
        if (iter % logInterval == 0) {
            std::cout << "Iteration " << iter << " - Best fitness so far: " << bestFitness << std::endl;
        }
    }
    
    std::cout << "Best solution found: ";
    for (const auto& var : bestSolution) std::cout << var << " ";
    std::cout << "\nObjective function value at best solution: " << bestFitness << "\n";
    return bestFitness;
}

int main(int argc, char *argv[]) {
    // Default values for Harmony Search parameters
    int memorySize = 1000;
    int maxIterations = 100000;
    double harmonyMemoryConsideringRate = 0.9;
    double pitchAdjustingRate = 0.3;
    double bandwidth = 0.01;
    int logInterval = 1000;
    int dimensions = 2; // Default number of dimensions
    int seed = rd(); // default to a random seed

    // Read parameters from command line if provided
    try {
        if (argc > 1) memorySize = std::stoi(argv[1]);
        if (argc > 2) maxIterations = std::stoi(argv[2]);
        if (argc > 3) harmonyMemoryConsideringRate = std::stod(argv[3]);
        if (argc > 4) pitchAdjustingRate = std::stod(argv[4]);
        if (argc > 5) bandwidth = std::stod(argv[5]);
        if (argc > 6) logInterval = std::stoi(argv[6]);
        if (argc > 7) dimensions = std::stoi(argv[7]);
        if (argc > 8) seed = std::stoi(argv[8]);
    } catch (const std::invalid_argument& e) {
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
    std::cout << "log Interval = " << logInterval << std::endl;
    std::cout << "dimensions = " << dimensions << std::endl;
    std::cout << "seed = " << seed << std::endl;

    // Select and run objective functions with variable dimensions
    std::cout << "\nRunning with Michalewicz function\n";
    auto start = std::chrono::high_resolution_clock::now();
    harmonySearch(memorySize, harmonyMemoryConsideringRate, pitchAdjustingRate, bandwidth, maxIterations, dimensions, michalewicz, logInterval);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    std::cout << "\nRunning with Rosenbrock function\n";
    start = std::chrono::high_resolution_clock::now();
    harmonySearch(memorySize, harmonyMemoryConsideringRate, pitchAdjustingRate, bandwidth, maxIterations, dimensions, rosenbrock, logInterval);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "Execution time: " << duration.count() << " seconds\n";

    std::cout << "#####################################################################\n";

    return 0;
}
