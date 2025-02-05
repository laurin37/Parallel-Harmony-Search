// Harmony Search Algorithm Implementation in C++
// Improved to sort by harmony fitness and handle multiple harmony memory creation
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
#include <omp.h> 

/**
 * Clamps a value within the range [min, max].
 * @param value The value to clamp.
 * @param min The minimum allowed value.
 * @param max The maximum allowed value.
 * @return The clamped value.
 */
double clamp(double value, double min, double max) 
{
    return (value < min) ? min : (value > max ? max : value);
}

// Typedef for clarity
typedef std::vector<double> Solution;
typedef std::function<double(const Solution&)> ObjectiveFunction;

/**
 * Abstract base class for defining objective functions.
 * Any specific objective function must implement the evaluate method.
 */
class ObjectiveFunctionBase 
{
public:
    virtual ~ObjectiveFunctionBase() = default;

    /**
     * Evaluates the objective function for a given solution.
     * @param sol The solution to evaluate.
     * @return The computed fitness value.
     */
    virtual double evaluate(const Solution& sol) const = 0;
};

/**
 * Implementation of the Rosenbrock function as an objective function.
 * The Rosenbrock function is commonly used to evaluate optimization algorithms.
 */
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

/**
 * Generates random numbers within specified ranges.
 */
class RandomGenerator 
{
public:
    RandomGenerator(unsigned int seed) : gen(seed) {}

    /**
     * Generates a random double in the range [min, max].
     */
    double getDouble(double min, double max) 
    {
        std::uniform_real_distribution<> dis(min, max);
        return dis(gen);
    }

    /**
     * Generates a random integer in the range [min, max].
     */
    int getInt(int min, int max) 
    {
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }

private:
    std::mt19937 gen;
};

/**
 * Implements the Harmony Search algorithm.
 * This class manages the optimization process using Harmony Search.
 */
class HarmonySearch 
{
public:
    HarmonySearch(int dimensions, int hms, double hmcr, double par, double bw, int maxIter, 
                  const ObjectiveFunctionBase& objFunc, const Solution& lowerBounds, const Solution& upperBounds, unsigned int seed)
        : dimensions(dimensions), hms(hms), hmcr(hmcr), par(par), bw(bw), maxIter(maxIter),
          objectiveFunction(objFunc), lowerBounds(lowerBounds), upperBounds(upperBounds), rng(seed) 
    {
        if (hms <= 0) throw std::invalid_argument("Harmony memory size (hms) must be greater than 0.");
        if (hmcr < 0.0 || hmcr > 1.0) throw std::invalid_argument("HMCR must be in [0, 1].");
        if (par < 0.0 || par > 1.0) throw std::invalid_argument("PAR must be in [0, 1].");
        if (bw <= 0.0) throw std::invalid_argument("Bandwidth (bw) must be greater than 0.");
    }
    void initializeHarmonyMemory() {
        harmonyMemory.resize(hms);
        fitness.resize(hms);

        for (int i = 0; i < hms; ++i) {
            harmonyMemory[i] = randomSolution();
            fitness[i] = objectiveFunction.evaluate(harmonyMemory[i]);
        }

        // Sort HM based on fitness
        std::vector<std::pair<double, Solution>> sorted;
        for (int i = 0; i < hms; ++i) {
            sorted.emplace_back(fitness[i], harmonyMemory[i]);
        }
        // added sorting functionality
        /**********************/
        std::sort(sorted.begin(), sorted.end(), 
          [](const std::pair<double, Solution>& a, const std::pair<double, Solution>& b) { 
              return a.first < b.first; 
          });

        for (int i = 0; i < hms; ++i) {
            fitness[i] = sorted[i].first;
            harmonyMemory[i] = sorted[i].second;
        }
        /**********************/


        bestSolution = harmonyMemory[0];
        bestFitness = fitness[0];
        worstFitness = fitness.back();
        worstIndex = hms - 1;
    }
    // Compute the HM Search
    void runIterations(int numIterations) {
        for (int iter = 0; iter < numIterations; ++iter) {
            Solution newHarmony = generateNewHarmony();
            double newFitness = objectiveFunction.evaluate(newHarmony);

            if (newFitness < worstFitness) {
                replaceWorstHarmony(newHarmony, newFitness);
            }
        }
    }
    void replaceFirstHarmony(const Solution& newBest, double newFitness) {
        harmonyMemory[0] = newBest;
        fitness[0] = newFitness;

        // Update best and worst
        bestFitness = fitness[0];
        bestSolution = harmonyMemory[0];
        worstFitness = fitness[0];
        worstIndex = 0;

        for (int i = 1; i < hms; ++i) {
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

    Solution getBestSolution() const { return bestSolution; }
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
    double executionTimeInit = 0.0;
    double executionTimeOpt = 0.0;

    
    /**
     * Generates a random solution within the specified bounds.
     * @return A random solution.
     */
    Solution randomSolution() 
    {
        Solution solution(dimensions);
        for (int d = 0; d < dimensions; ++d) 
        {
            solution[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
        }
        return solution;
    }

    /**
     * Generates a new harmony based on the Harmony Memory and randomness.
     * @return The new harmony (solution).
     */
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
            } 
            else 
            {
                newHarmony[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
            }
        }

        return newHarmony;
    }
    
    /**
     * Replaces the worst harmony in the Harmony Memory if the new harmony is better.
     * @param newHarmony The new harmony to insert.
     * @param newFitness The fitness of the new harmony.
     */
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

/**
 * Writes optimization results to a CSV file.
 * If the file does not exist, a header is created.
 * @param filename Name of the CSV file.
 * @param dimensions Number of dimensions in the problem.
 * @param hms Harmony memory size.
 * @param hmcr Harmony memory consideration rate.
 * @param par Pitch adjustment rate.
 * @param bw Bandwidth for pitch adjustment.
 * @param maxIter Maximum iterations.
 * @param executionTime Execution time of the optimization.
 * @param numCores Number of cores used.
 * @param seed Random seed used for reproducibility.
 * @param bestFitness The best fitness value found.
 * @param executionType The execution type (e.g., Sequential).
 */
void writeResultsToCSV(const std::string& filename, int dimensions, int hms, double hmcr, double par, double bw, int maxIter, 
                       double executionTime, int numCores, unsigned int seed, double bestFitness, const std::string& executionType) 
{
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
int main(int argc, char* argv[]) 
{
    if (argc != 9) 
    {
        std::cerr << "Usage: " << argv[0] << " <hms> <hmcr> <par> <bw> <maxIter> <dimensions> <seed> <thread_size>\n";
        return 1;
    }

    // Read and validate parameters
    const int hms = std::stoi(argv[1]);
    const double hmcr = std::stod(argv[2]);
    const double par = std::stod(argv[3]);
    const double bw = std::stod(argv[4]);
    const int maxIter = std::stoi(argv[5]);
    const int dimensions = std::stoi(argv[6]);
    const unsigned int seed = std::stoul(argv[7]);
    const unsigned int num_threads = std::stoul(argv[8]);
    omp_set_num_threads(num_threads);

    try {
        RosenbrockFunction rosenbrock;
        const Solution lowerBounds(dimensions, -5.0);
        const Solution upperBounds(dimensions, 5.0);

        // create a harmonysearch object for each thread
        std::vector<HarmonySearch> harmonies;
        harmonies.reserve(num_threads);
        for (unsigned int i = 0; i < num_threads; ++i) {
            harmonies.emplace_back(dimensions, hms, hmcr, par, bw, maxIter, 
                                  rosenbrock, lowerBounds, upperBounds, seed + i);
        }

        auto start = std::chrono::high_resolution_clock::now();

        // Initialize harmony memory in parallel - use static as default will introduce overhead(load is already balanced)
        #pragma omp parallel for schedule(static)
        for (unsigned int i = 0; i < num_threads; ++i) {
            harmonies[i].initializeHarmonyMemory();
        }

        int parallelIterations = 0;
        const int IK = maxIter / num_threads; 
        
        Solution globalBest;
        double globalBestFitness = std::numeric_limits<double>::infinity();
        #pragma omp parallel
        {
            while (parallelIterations < maxIter) {
                int currentIK;
                
                // Single thread manages iteration control
                #pragma omp single
                {
                    const int remaining = maxIter - parallelIterations;
                    currentIK = std::min(IK, remaining);
                }

                // Parallel execution of iterations
                #pragma omp for schedule(static) nowait
                for (unsigned int i = 0; i < num_threads; ++i) {
                    harmonies[i].runIterations(currentIK);
                }

                // Find global best - single thread reduces results
                #pragma omp single
                {
                    globalBestFitness = harmonies[0].getBestFitness();
                    globalBest = harmonies[0].getBestSolution();
                    for (unsigned int i = 1; i < num_threads; ++i) {
                        if (harmonies[i].getBestFitness() < globalBestFitness) {
                            globalBestFitness = harmonies[i].getBestFitness();
                            globalBest = harmonies[i].getBestSolution();
                        }
                    }
                }

                // Parallel update of all harmony memories
                #pragma omp for schedule(static)
                for (unsigned int i = 0; i < num_threads; ++i) {
                    harmonies[i].replaceFirstHarmony(globalBest, globalBestFitness);
                }

                // Single thread updates iteration counter
                #pragma omp single
                {
                    parallelIterations += currentIK * num_threads;
                }
            }
        } 

        // Timing and results output
        const auto end = std::chrono::high_resolution_clock::now();
        const double executionTime = 
            std::chrono::duration<double>(end - start).count();

        // Use the final globalBest instead of re-searching
        std::cout << "\n==================== Parallel Run Start ====================\n"
                  << "Cores: " << num_threads << "\n"
                  << "Dimensions: " << dimensions << "\n"
                  << "HMS: " << hms << ", HMCR: " << hmcr << ", PAR: " << par 
                  << ", BW: " << bw << "\n"
                  << "Max Iterations: " << maxIter << ", Seed: " << seed << "\n"
                  << "Execution Time: " << executionTime << " seconds\n"
                  << "Best fitness: " << harmonies[0].getBestFitness() << "\n"
                  << "==================== Run End ======================\n";

        writeResultsToCSV("harmony_search_results.csv", dimensions, hms, hmcr, par, bw, maxIter,
                         executionTime, num_threads, seed, harmonies[0].getBestFitness(), "PHS1");
    } 
    catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch (...) {
        std::cerr << "Unknown error occurred\n";
        return 2;
    }

    return 0;
}
