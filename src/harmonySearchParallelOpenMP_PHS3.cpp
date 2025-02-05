// PHS Without Communication (PHS3) Implementation in C++
// Each Harmony Search instance runs independently; no migration/sharing among threads.

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
 * (No changes from a typical HS, except we won't do any cross-thread sharing.)
 */
class HarmonySearch
{
public:
    HarmonySearch(int dimensions, int hms, double hmcr, double par, double bw, int maxIter,
                  const ObjectiveFunctionBase& objFunc,
                  const Solution& lowerBounds,
                  const Solution& upperBounds,
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
     * Initializes the harmony memory with random solutions and sorts them.
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
     * Runs the entire Harmony Search for maxIter iterations.
     */
    void run()
    {
        for (int iter = 0; iter < maxIter; ++iter)
        {
            // Generate a new candidate solution
            Solution newSol  = generateNewHarmony();
            double   newFit  = objective.evaluate(newSol);

            // If it's better than the worst in HM, replace
            if (newFit < worstFitness)
            {
                harmonyMemory[worstIndex] = newSol;
                fitness[worstIndex]       = newFit;

                // Update the worst index/fitness
                worstIndex   = std::distance(fitness.begin(),
                                             std::max_element(fitness.begin(), fitness.end()));
                worstFitness = fitness[worstIndex];

                // Possibly update the best
                if (newFit < bestFitness)
                {
                    bestSolution = newSol;
                    bestFitness  = newFit;
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
     * Sorts the harmony memory by ascending fitness.
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
     * Updates bestFitness, bestSolution, worstFitness, worstIndex after sorting.
     */
    void updateBestWorst()
    {
        bestFitness  = fitness[0];
        bestSolution = harmonyMemory[0];

        worstFitness = fitness.back();
        worstIndex   = hms - 1;
    }

    /**
     * Generates a new harmony from HM (with pitch adjustment) or random.
     */
    Solution generateNewHarmony()
    {
        Solution newSol(dimensions);

        for (int d = 0; d < dimensions; ++d)
        {
            if (rng.getDouble(0.0, 1.0) < hmcr)
            {
                // pick existing dimension from HM
                int index     = rng.getInt(0, hms - 1);
                double value  = harmonyMemory[index][d];

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
 * Write results to CSV (optional).
 */
void writeResultsToCSV(const std::string& filename,
                       int dimensions, int hms, double hmcr, double par, double bw,
                       int maxIter, double executionTime, int numCores, unsigned int seed,
                       double bestFitness, const std::string& execType)
{
    std::ofstream file;
    // check if file already exists
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
                  << " <HMS> <HMCR> <PAR> <BW> <maxIter> <dimensions> <seed> <numThreads>\n";
        return 1;
    }

    // Parse command-line arguments
    const int    hms        = std::stoi(argv[1]);
    const double hmcr       = std::stod(argv[2]);
    const double par        = std::stod(argv[3]);
    const double bw         = std::stod(argv[4]);
    const int    dimensions = std::stoi(argv[6]);
    const unsigned int seed        = static_cast<unsigned int>(std::stoul(argv[7]));
    const unsigned int numThreads  = static_cast<unsigned int>(std::stoul(argv[8]));

    // Set the number of threads for OpenMP
    omp_set_num_threads(numThreads);

    const int    maxIter    = std::stoi(argv[5])/numThreads;  // divide iterations among threads
    // Create the objective function & bounds
    RosenbrockFunction rosenbrock;
    Solution lower(dimensions, -5.0);
    Solution upper(dimensions,  5.0);

    try
    {
        // Create one HarmonySearch object per thread
        std::vector<HarmonySearch> allSearches;
        allSearches.reserve(numThreads);

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            // Slightly vary the seed per thread
            allSearches.emplace_back(dimensions, hms, hmcr, par, bw, maxIter,
                                     rosenbrock, lower, upper, seed + t);
        }

        auto startTime = std::chrono::high_resolution_clock::now();

        // Run all HS in parallel, with no communication
        #pragma omp parallel for
        for (unsigned int t = 0; t < numThreads; ++t)
        {
            allSearches[t].initialize();   // initialize HM
            allSearches[t].run();          // run full maxIter
        }

        // After parallel search, find the best among all
        double   globalBestFitness = std::numeric_limits<double>::infinity();
        Solution globalBest;

        for (unsigned int t = 0; t < numThreads; ++t)
        {
            double f = allSearches[t].getBestFitness();
            if (f < globalBestFitness)
            {
                globalBestFitness = f;
                globalBest        = allSearches[t].getBestSolution();
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double executionTime =
            std::chrono::duration<double>(endTime - startTime).count();

        // Print results
        std::cout << "\n=========== PHS Without Communication (PHS3) ===========\n"
                  << "Threads:       " << numThreads   << "\n"
                  << "Dimensions:    " << dimensions   << "\n"
                  << "HMS: " << hms << ", HMCR: " << hmcr << ", PAR: " << par
                  << ", BW: " << bw << "\n"
                  << "Max Iter:      " << maxIter      << "\n"
                  << "Seed:          " << seed         << "\n"
                  << "ExecutionTime: " << executionTime << " seconds\n"
                  << "BestFitness:   " << globalBestFitness << "\n"
                  << "======================================================\n";

        // Write optional CSV
        writeResultsToCSV("phs3_results.csv", dimensions, hms, hmcr, par, bw,
                          maxIter, executionTime, numThreads, seed,
                          globalBestFitness, "PHS3");
    }
    catch(const std::invalid_argument& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    catch(...)
    {
        std::cerr << "Unknown error occurred.\n";
        return 2;
    }

    return 0;
}
