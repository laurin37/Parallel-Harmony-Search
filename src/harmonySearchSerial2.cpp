// Harmony Search Algorithm Implementation in C++

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>
#include <limits>
#include <chrono>
#include <cstdlib> // For command-line argument parsing

// Clamps a value within the range [min, max]
double clamp(double value, double min, double max) 
{
    return (value < min) ? min : (value > max ? max : value);
}

// Typedef for clarity
typedef std::vector<double> Solution;
typedef std::function<double(const Solution&)> ObjectiveFunction;

// Random number generator
class RandomGenerator {
public:
    RandomGenerator(int seed = 42) : gen(seed) {}

    double getDouble(double min, double max) {
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

    int getInt(int min, int max) {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

private:
    std::random_device rd;
    std::mt19937 gen;
};

// Harmony Search Algorithm class
class HarmonySearch {
public:
    HarmonySearch(int dimensions, int hms, double hmcr, double par, double bw, int maxIter, 
                  const ObjectiveFunction& objFunc, const Solution& lowerBounds, const Solution& upperBounds)
        : dimensions(dimensions), hms(hms), hmcr(hmcr), par(par), bw(bw), maxIter(maxIter),
          objectiveFunction(objFunc), lowerBounds(lowerBounds), upperBounds(upperBounds), rng() {}

    Solution optimize() {
        auto start = std::chrono::high_resolution_clock::now();        
        initializeHarmonyMemory();

        for (int iter = 0; iter < maxIter; ++iter) {
            Solution newHarmony = generateNewHarmony();
            double newFitness = objectiveFunction(newHarmony);

            if (newFitness < worstFitness) {
                replaceWorstHarmony(newHarmony, newFitness);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " seconds\n";

        return bestSolution;
    }

private:
    int dimensions;
    int hms; // Harmony memory size
    double hmcr; // Harmony memory consideration rate
    double par;  // Pitch adjustment rate
    double bw;   // Bandwidth for pitch adjustment
    int maxIter; // Maximum iterations
    ObjectiveFunction objectiveFunction;
    Solution lowerBounds;
    Solution upperBounds;
    RandomGenerator rng;

    std::vector<Solution> harmonyMemory;
    std::vector<double> fitness;
    Solution bestSolution;
    double bestFitness = std::numeric_limits<double>::infinity();
    double worstFitness = -std::numeric_limits<double>::infinity();
    int worstIndex = 0;

    void initializeHarmonyMemory() {
        harmonyMemory.resize(hms);
        fitness.resize(hms);

        for (int i = 0; i < hms; ++i) {
            harmonyMemory[i] = randomSolution();
            fitness[i] = objectiveFunction(harmonyMemory[i]);

            if (fitness[i] < bestFitness) {
                bestFitness = fitness[i];
                bestSolution = harmonyMemory[i];
            }

            if (fitness[i] > worstFitness) {
                worstFitness = fitness[i];
                worstIndex = i;
            }
        }
    }

    Solution randomSolution() {
        Solution solution(dimensions);
        for (int d = 0; d < dimensions; ++d) {
            solution[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
        }
        return solution;
    }

    Solution generateNewHarmony() {
        Solution newHarmony(dimensions);

        for (int d = 0; d < dimensions; ++d) {
            if (rng.getDouble(0.0, 1.0) < hmcr) {
                newHarmony[d] = harmonyMemory[rng.getInt(0, hms - 1)][d];

                if (rng.getDouble(0.0, 1.0) < par) {
                    newHarmony[d] += rng.getDouble(-bw, bw);
                    newHarmony[d] = clamp(newHarmony[d], lowerBounds[d], upperBounds[d]);
                }
            } else {
                newHarmony[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
            }
        }

        return newHarmony;
    }

    void replaceWorstHarmony(const Solution& newHarmony, double newFitness) {
        harmonyMemory[worstIndex] = newHarmony;
        fitness[worstIndex] = newFitness;

        worstIndex = std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end()));
        worstFitness = fitness[worstIndex];

        if (newFitness < bestFitness) {
            bestFitness = newFitness;
            bestSolution = newHarmony;
        }
    }
};

// Example usage
int main(int argc, char* argv[]) {
    std::cout << "\n==================== Run Start ====================\n";

    // Harmony Search default parameters
    int hms = 10000;
    double hmcr = 0.9; //0.7 - 0.95
    double par = 0.3; //0.1 - 0.5
    double bw = 0.01;
    int maxIter = 1000000;

    // Read parameters from command line if provided
    try 
    {
        if (argc > 1) hms = std::stoi(argv[1]);
        if (argc > 2) hmcr = std::stod(argv[2]);
        if (argc > 3) par = std::stod(argv[3]);
        if (argc > 4) bw = std::stod(argv[4]);
        if (argc > 5) maxIter = std::stoi(argv[5]);
    } catch (const std::invalid_argument& e) 
    {
        std::cerr << "Invalid argument: please enter numeric values for all parameters." << std::endl;
        return 1;
    }

    // Print parameter values
    std::cout << "memory Size = " << hms << std::endl;
    std::cout << "max Iterations = " << maxIter << std::endl;
    std::cout << "harmony Memory Considering Rate = " << hmcr << std::endl;
    std::cout << "pitch Adjusting Rate = " << par << std::endl;
    std::cout << "band width = " << bw << std::endl;

    // Define the Rosenbrock function
    ObjectiveFunction rosenbrock = [](const Solution& sol) {
        double sum = 0.0;
        for (size_t i = 0; i < sol.size() - 1; ++i) {
            double term1 = (sol[i + 1] - sol[i] * sol[i]);
            double term2 = (1.0 - sol[i]);
            sum += 100.0 * term1 * term1 + term2 * term2;
        }
        return sum;
    };

    // Problem parameters
    int dimensions = 5; // Define the dimension size
    Solution lowerBounds(dimensions, -5.0);
    Solution upperBounds(dimensions, 5.0);

    HarmonySearch hs(dimensions, hms, hmcr, par, bw, maxIter, rosenbrock, lowerBounds, upperBounds);
    Solution best = hs.optimize();

    std::cout << "Best solution found:" << std::endl;
    for (double x : best) {
        std::cout << x << " ";
    }
    std::cout << "\nBest fitness: " << rosenbrock(best) << std::endl;
    std::cout << "==================== Run End ======================\n";

    return 0;
}
