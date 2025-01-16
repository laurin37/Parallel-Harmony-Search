// Harmony Search Algorithm Implementation in C++

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>
#include <limits>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <sstream>
#include <unordered_map>

// Clamps a value within the range [min, max]
double clamp(double value, double min, double max) 
{
    return (value < min) ? min : (value > max ? max : value);
}

// Typedef for clarity
typedef std::vector<double> Solution;
typedef std::function<double(const Solution&)> ObjectiveFunction;

// Abstract base class for objective functions
class ObjectiveFunctionBase 
{
public:
    virtual ~ObjectiveFunctionBase() = default;
    virtual double evaluate(const Solution& sol) const = 0;
};

// Rosenbrock function implementation
class RosenbrockFunction : public ObjectiveFunctionBase 
{
public:
    double evaluate(const Solution& sol) const override 
    {
        double sum = 0.0;
        for (size_t i = 0; i < sol.size() - 1; ++i) 
        {
            double term1 = (sol[i + 1] - sol[i] * sol[i]);
            double term2 = (1.0 - sol[i]);
            sum += 100.0 * term1 * term1 + term2 * term2;
        }
        return sum;
    }
};

// Random number generator
class RandomGenerator 
{
public:
    RandomGenerator(unsigned int seed) : gen(seed) {}

    double getDouble(double min, double max) 
    {
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

    int getInt(int min, int max) 
    {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

private:
    std::mt19937 gen;
};

// Harmony Search Algorithm class
class HarmonySearch 
{
public:
    HarmonySearch(int dimensions, int hms, double hmcr, double par, double bw, int maxIter, 
                  const ObjectiveFunctionBase& objFunc, const Solution& lowerBounds, const Solution& upperBounds, unsigned int seed)
        : dimensions(dimensions), hms(hms), hmcr(hmcr), par(par), bw(bw), maxIter(maxIter),
          objectiveFunction(objFunc), lowerBounds(lowerBounds), upperBounds(upperBounds), rng(seed) {}

    Solution optimize() 
    {
        auto start = std::chrono::high_resolution_clock::now();        
        initializeHarmonyMemory();

        for (int iter = 0; iter < maxIter; ++iter) 
        {
            Solution newHarmony = generateNewHarmony();
            double newFitness = objectiveFunction.evaluate(newHarmony);

            if (newFitness < worstFitness) 
            {
                replaceWorstHarmony(newHarmony, newFitness);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        executionTime = duration.count();
        return bestSolution;
    }

    double getExecutionTime() const { return executionTime; }
    double getBestFitness() const { return bestFitness; }

private:
    int dimensions;
    int hms; // Harmony memory size
    double hmcr; // Harmony memory consideration rate
    double par;  // Pitch adjustment rate
    double bw;   // Bandwidth for pitch adjustment
    int maxIter; // Maximum iterations
    const ObjectiveFunctionBase& objectiveFunction;
    Solution lowerBounds;
    Solution upperBounds;
    RandomGenerator rng;

    std::vector<Solution> harmonyMemory;
    std::vector<double> fitness;
    Solution bestSolution;
    double bestFitness = std::numeric_limits<double>::infinity();
    double worstFitness = -std::numeric_limits<double>::infinity();
    int worstIndex = 0;
    double executionTime = 0.0;

    void initializeHarmonyMemory() 
    {
        harmonyMemory.resize(hms);
        fitness.resize(hms);

        for (int i = 0; i < hms; ++i) 
        {
            harmonyMemory[i] = randomSolution();
            fitness[i] = objectiveFunction.evaluate(harmonyMemory[i]);

            if (fitness[i] < bestFitness) 
            {
                bestFitness = fitness[i];
                bestSolution = harmonyMemory[i];
            }

            if (fitness[i] > worstFitness) 
            {
                worstFitness = fitness[i];
                worstIndex = i;
            }
        }
    }

    Solution randomSolution() 
    {
        Solution solution(dimensions);
        for (int d = 0; d < dimensions; ++d) 
        {
            solution[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
        }
        return solution;
    }

    Solution generateNewHarmony() 
    {
        Solution newHarmony(dimensions);

        for (int d = 0; d < dimensions; ++d) 
        {
            if (rng.getDouble(0.0, 1.0) < hmcr) 
            {
                newHarmony[d] = harmonyMemory[rng.getInt(0, hms - 1)][d];

                if (rng.getDouble(0.0, 1.0) < par) 
                {
                    newHarmony[d] += rng.getDouble(-bw, bw);
                    newHarmony[d] = clamp(newHarmony[d], lowerBounds[d], upperBounds[d]);
                }
            } else 
            {
                newHarmony[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
            }
        }

        return newHarmony;
    }

    void replaceWorstHarmony(const Solution& newHarmony, double newFitness) 
    {
        harmonyMemory[worstIndex] = newHarmony;
        fitness[worstIndex] = newFitness;

        worstIndex = std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end()));
        worstFitness = fitness[worstIndex];

        if (newFitness < bestFitness) 
        {
            bestFitness = newFitness;
            bestSolution = newHarmony;
        }
    }
};

// Helper function to append or create a CSV file
void writeResultsToCSV(const std::string& filename, int dimensions, int hms, double hmcr, double par, double bw, int maxIter, 
                       double executionTime, int numCores, unsigned int seed, double bestFitness, const std::string& executionType) {
    std::ofstream file;
    bool fileExists = std::ifstream(filename).good();

    file.open(filename, std::ios::app);

    if (!fileExists) 
    {
        file << "Dimensions,HMS,HMCR,PAR,BW,MaxIter,ExecutionTime(s),Cores,Seed,BestFitness,ExecutionType\n";
    }

    file << dimensions << "," << hms << "," << hmcr << "," << par << "," << bw << "," << maxIter << "," 
         << executionTime << "," << numCores << "," << seed << "," << bestFitness << "," << executionType << "\n";

    file.close();
}

// Example usage
int main(int argc, char* argv[]) 
{
    if (argc != 8) 
    {
        std::cerr << "Usage: " << argv[0] << " <hms> <hmcr> <par> <bw> <maxIter> <dimensions> <seed>\n";
        return 1;
    }

    // Parse command-line arguments
    int hms = std::stoi(argv[1]);
    double hmcr = std::stod(argv[2]);
    double par = std::stod(argv[3]);
    double bw = std::stod(argv[4]);
    int maxIter = std::stoi(argv[5]);
    int dimensions = std::stoi(argv[6]);
    unsigned int seed = std::stoul(argv[7]);

    // Define the Rosenbrock function
    RosenbrockFunction rosenbrock;

    // Problem parameters
    Solution lowerBounds(dimensions, -5.0);
    Solution upperBounds(dimensions, 5.0);

    HarmonySearch hs(dimensions, hms, hmcr, par, bw, maxIter, rosenbrock, lowerBounds, upperBounds, seed);
    Solution best = hs.optimize();

    // Output formatting
    std::cout << "\n==================== Run Start ====================\n";
    std::cout << "Dimensions: " << dimensions << "\n";
    std::cout << "HMS: " << hms << ", HMCR: " << hmcr << ", PAR: " << par << ", BW: " << bw << "\n";
    std::cout << "Max Iterations: " << maxIter << ", Seed: " << seed << "\n";
    std::cout << "Execution Time: " << hs.getExecutionTime() << " seconds\n";
    std::cout << "Best fitness: " << hs.getBestFitness() << std::endl;
    std::cout << "==================== Run End ======================" << std::endl;

    // Write results to CSV
    writeResultsToCSV("harmony_search_results.csv", dimensions, hms, hmcr, par, bw, maxIter, hs.getExecutionTime(), 
                      1, seed, hs.getBestFitness(), "Sequential");

    return 0;
}
