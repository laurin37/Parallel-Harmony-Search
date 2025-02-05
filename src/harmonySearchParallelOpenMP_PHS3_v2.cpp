#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>
#include <limits>
#include <chrono>
#include <fstream>
#include <omp.h>

/**
 * Clamps a value within the range [minVal, maxVal].
 */
double clamp(double value, double minVal, double maxVal)
{
    return (value < minVal) ? minVal : (value > maxVal ? maxVal : value);
}

typedef std::vector<double> Solution;

/**
 * Abstract base class for defining objective functions.
 */
class ObjectiveFunctionBase
{
public:
    virtual ~ObjectiveFunctionBase() = default;
    virtual double evaluate(const Solution& sol) const = 0;
};

/**
 * Example: Rosenbrock function.
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
 * Random number generator utility.
 */
class RandomGenerator
{
public:
    RandomGenerator(unsigned int seed) : gen(seed) {}

    double getDouble(double minVal, double maxVal)
    {
        std::uniform_real_distribution<double> dist(minVal, maxVal);
        return dist(gen);
    }

    int getInt(int minVal, int maxVal)
    {
        std::uniform_int_distribution<int> dist(minVal, maxVal);
        return dist(gen);
    }

private:
    std::mt19937 gen;
};

/**
 * Harmony Search Implementation
 */
class HarmonySearch
{
public:
    HarmonySearch(int dimensions, int hms, double hmcr, double par, double bw,
                  int maxIter, const ObjectiveFunctionBase& objFunc,
                  const Solution& lowerBounds, const Solution& upperBounds,
                  unsigned int seed)
        : dimensions(dimensions), hms(hms), hmcr(hmcr), par(par), bw(bw),
          maxIter(maxIter), objective(objFunc),
          lowerBounds(lowerBounds), upperBounds(upperBounds), rng(seed)
    {
        if (hms <= 0) throw std::invalid_argument("Harmony memory size must be > 0.");
        if (hmcr < 0.0 || hmcr > 1.0) throw std::invalid_argument("HMCR must be in [0, 1].");
        if (par < 0.0 || par > 1.0) throw std::invalid_argument("PAR must be in [0, 1].");
        if (bw <= 0.0) throw std::invalid_argument("Bandwidth (bw) must be > 0.");
    }

    /**
     * Initialize the harmony memory with random solutions and sort them.
     */
    void initialize()
    {
        harmonyMemory.resize(hms);
        fitness.resize(hms);

        for (int i = 0; i < hms; ++i)
        {
            harmonyMemory[i] = randomSolution();
            fitness[i]       = objective.evaluate(harmonyMemory[i]);
        }

        sortHarmonyMemory();
        updateBestWorst();
    }

    /**
     * Perform a given number of HS iterations (rather than the entire maxIter).
     * This allows us to do partial/batch runs and then do communication in-between.
     */
    void runIterations(int numIters)
    {
        for (int iter = 0; iter < numIters; ++iter)
        {
            // Generate a new candidate
            Solution newSol = generateNewHarmony();
            double   newFit = objective.evaluate(newSol);

            // If it's better than the worst in HM, replace
            if (newFit < worstFitness)
            {
                harmonyMemory[worstIndex] = newSol;
                fitness[worstIndex]       = newFit;

                // Update worst
                worstIndex   = std::distance(fitness.begin(),
                                             std::max_element(fitness.begin(), fitness.end()));
                worstFitness = fitness[worstIndex];

                // Possibly update best
                if (newFit < bestFitness)
                {
                    bestFitness  = newFit;
                    bestSolution = newSol;
                }
            }
        }
    }

    /**
     * Insert one or more external solutions (e.g., from other threads) into
     * the local memory, if they are better than local worst solutions.
     *
     * For example, we replace the worst solutions with these external bests.
     */
    void insertExternalSolutions(const std::vector<Solution>& solutions)
    {
        for (const auto& extSol : solutions)
        {
            double extFit = objective.evaluate(extSol);
            // If it's better than the local worst, replace
            if (extFit < worstFitness)
            {
                harmonyMemory[worstIndex] = extSol;
                fitness[worstIndex]       = extFit;

                // Re-find the worst
                worstIndex = std::distance(fitness.begin(),
                                           std::max_element(fitness.begin(), fitness.end()));
                worstFitness = fitness[worstIndex];

                // Possibly update local best
                if (extFit < bestFitness)
                {
                    bestFitness  = extFit;
                    bestSolution = extSol;
                }
            }
        }
    }

    double   getBestFitness()  const { return bestFitness;  }
    Solution getBestSolution() const { return bestSolution; }

private:
    int dimensions;
    int hms;
    double hmcr;
    double par;
    double bw;
    int maxIter;
    const ObjectiveFunctionBase& objective;
    Solution lowerBounds;
    Solution upperBounds;
    RandomGenerator rng;

    std::vector<Solution> harmonyMemory;
    std::vector<double>   fitness;

    Solution bestSolution;
    double   bestFitness   = std::numeric_limits<double>::infinity();
    double   worstFitness  = -std::numeric_limits<double>::infinity();
    int      worstIndex    = 0;

private:
    /**
     * Generates a random feasible solution.
     */
    Solution randomSolution()
    {
        Solution sol(dimensions);
        for (int d = 0; d < dimensions; ++d)
        {
            sol[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
        }
        return sol;
    }

    /**
     * Sort the memory by ascending fitness.
     */
    void sortHarmonyMemory()
    {
        std::vector<std::pair<double, Solution>> temp;
        temp.reserve(hms);
        for (int i = 0; i < hms; ++i)
        {
            temp.emplace_back(fitness[i], harmonyMemory[i]);
        }

        std::sort(temp.begin(), temp.end(),
                  [](const std::pair<double, Solution>& a, const std::pair<double, Solution>& b) { return a.first < b.first; });

        for (int i = 0; i < hms; ++i)
        {
            fitness[i]       = temp[i].first;
            harmonyMemory[i] = temp[i].second;
        }
    }

    /**
     * Update best/worst references after sorting.
     */
    void updateBestWorst()
    {
        bestFitness  = fitness[0];
        bestSolution = harmonyMemory[0];

        worstFitness = fitness.back();
        worstIndex   = hms - 1;
    }

    /**
     * Generates a new harmony from HM or random, plus pitch adjustment.
     */
    Solution generateNewHarmony()
    {
        Solution newSol(dimensions);

        for (int d = 0; d < dimensions; ++d)
        {
            if (rng.getDouble(0.0, 1.0) < hmcr)
            {
                // pick from memory
                int index    = rng.getInt(0, hms - 1);
                double value = harmonyMemory[index][d];

                // pitch adjustment
                if (rng.getDouble(0.0, 1.0) < par)
                {
                    value += rng.getDouble(-bw, bw);
                }
                newSol[d] = clamp(value, lowerBounds[d], upperBounds[d]);
            }
            else
            {
                // random dimension
                newSol[d] = rng.getDouble(lowerBounds[d], upperBounds[d]);
            }
        }

        return newSol;
    }
};

/**
 * Optional: Write results to CSV for logging.
 */
void writeResultsToCSV(const std::string& filename,
                       int dimensions, int hms, double hmcr, double par, double bw,
                       int maxIter, double executionTime, int numCores, unsigned int seed,
                       double bestFitness, const std::string& execType)
{
    std::ofstream file;
    bool fileExists = static_cast<bool>(std::ifstream(filename));
    file.open(filename, std::ios::app);

    if (!fileExists)
    {
        file << "Dimensions,HMS,HMCR,PAR,BW,MaxIter,ExecTime,Cores,Seed,BestFitness,ExecType\n";
    }

    file << dimensions << "," << hms << "," << hmcr << "," << par << "," << bw << ","
         << maxIter << "," << executionTime << "," << numCores << "," << seed << ","
         << bestFitness << "," << execType << "\n";

    file.close();
}


int main(int argc, char* argv[])
{
    if (argc != 9)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <HMS> <HMCR> <PAR> <BW> <maxIter> <dimensions> <seed> <numThreads> <batchSize>\n";
        return 1;
    }

    // Parse command-line arguments
    const int    hms        = std::stoi(argv[1]);
    const double hmcr       = std::stod(argv[2]);
    const double par        = std::stod(argv[3]);
    const double bw         = std::stod(argv[4]);
    const int    maxIter    = std::stoi(argv[5]);
    const int    dimensions = std::stoi(argv[6]);
    const unsigned int seed       = static_cast<unsigned int>(std::stoul(argv[7]));
    const unsigned int numThreads = static_cast<unsigned int>(std::stoul(argv[8]));

    // Set OMP threads
    omp_set_num_threads(numThreads);

    // Objective function & bounds
    RosenbrockFunction rosenbrock;
    Solution lower(dimensions, -5.0);
    Solution upper(dimensions,  5.0);

    try
    {
        // Create one HarmonySearch per thread
        std::vector<HarmonySearch> hsList;
        hsList.reserve(numThreads);

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            hsList.emplace_back(dimensions, hms, hmcr, par, bw,
                                maxIter, rosenbrock, lower, upper, seed + t);
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        // Initialize all in parallel
        #pragma omp parallel for
        for (unsigned int t = 0; t < numThreads; ++t)
        {
            hsList[t].initialize();
        }

        int doneIterations = 0;
        const int batchSize = maxIter / numThreads;
        // Repeatedly run "batchSize" iterations, then communicate
        while (doneIterations < maxIter)
        {
            int remaining = maxIter - doneIterations;
            int currentBatch = (remaining < batchSize) ? remaining : batchSize;
            if (currentBatch <= 0) break;

            // Each thread runs for "currentBatch" iterations
            #pragma omp parallel for
            for (unsigned int t = 0; t < numThreads; ++t)
            {
                hsList[t].runIterations(currentBatch);
            }

            doneIterations += currentBatch;

            // Communicate: gather best from each thread
            // We'll gather them in a vector, so each thread's best is included.
            std::vector<Solution> bestCollection(numThreads);
            #pragma omp parallel for
            for (unsigned int t = 0; t < numThreads; ++t)
            {
                bestCollection[t] = hsList[t].getBestSolution();
            }

            // Now broadcast them to all threads
            // Each thread updates its memory using all best solutions (except maybe its own).
            #pragma omp parallel for
            for (unsigned int t = 0; t < numThreads; ++t)
            {
                // Insert best solutions from across all threads
                hsList[t].insertExternalSolutions(bestCollection);
            }
        }

        // After the loop, find the global best
        double   globalBestFitness = std::numeric_limits<double>::infinity();
        Solution globalBest;

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            double fit = hsList[t].getBestFitness();
            if (fit < globalBestFitness)
            {
                globalBestFitness = fit;
                globalBest        = hsList[t].getBestSolution();
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double execTime = std::chrono::duration<double>(endTime - startTime).count();

        // Print results
        std::cout << "\n===== Parallel HS with Batched Communication =====\n"
                  << "Threads:      " << numThreads << "\n"
                  << "Dimensions:   " << dimensions << "\n"
                  << "HMS: " << hms << ", HMCR: " << hmcr << ", PAR: " << par << ", BW: " << bw << "\n"
                  << "MaxIter:      " << maxIter << ", BatchSize: " << batchSize << "\n"
                  << "Seed:         " << seed << "\n"
                  << "ExecutionTime: " << execTime << " s\n"
                  << "BestFitness:   " << globalBestFitness << "\n"
                  << "==================================================\n";

        // Optional: record to CSV
        writeResultsToCSV("harmony_search_results.csv", dimensions, hms, hmcr, par, bw,
                          maxIter, execTime, numThreads, seed,
                          globalBestFitness, "PHS3_Batched");
    }
    catch (const std::invalid_argument& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown error occurred.\n";
        return 2;
    }

    return 0;
}
