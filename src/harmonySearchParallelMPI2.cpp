// MPI Implementation of Harmony Search Algorithm

#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <cmath>
#include <limits>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <numeric>

// Typedef for clarity
typedef std::vector<double> Solution;
typedef std::function<double(const Solution&)> ObjectiveFunction;

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
 * Implements the Harmony Search algorithm with MPI.
 * This class manages the optimization process using Harmony Search across multiple processes.
 */
class MPIHarmonySearch
{
public:
    MPIHarmonySearch(int dimensions, int hms, double hmcr, double par, double bw, int maxIter, 
                     const ObjectiveFunctionBase& objFunc, const Solution& lowerBounds, const Solution& upperBounds, unsigned int seed)
        : dimensions(dimensions), hms(hms), hmcr(hmcr), par(par), bw(bw), maxIter(maxIter),
          objectiveFunction(objFunc), lowerBounds(lowerBounds), upperBounds(upperBounds), rng(seed)
    {
        if (hms <= 0)
        {
            throw std::invalid_argument("Harmony memory size (hms) must be greater than 0.");
        }
        if (hmcr < 0.0 || hmcr > 1.0)
        {
            throw std::invalid_argument("Harmony memory consideration rate (hmcr) must be in the range [0, 1].");
        }
        if (par < 0.0 || par > 1.0)
        {
            throw std::invalid_argument("Pitch adjustment rate (par) must be in the range [0, 1].");
        }
        if (bw <= 0.0)
        {
            throw std::invalid_argument("Bandwidth (bw) must be greater than 0.");
        }
    }

    /**
     * Optimizes the objective function using Harmony Search with MPI.
     * @return The best solution and its fitness.
     */
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

        double globalFitness;
        MPI_Allreduce(&bestFitness, &globalFitness, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        bestFitness = globalFitness;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        executionTime = duration.count();
        return bestSolution;
    }

    double getExecutionTime() const { return executionTime; }
    double getBestFitness() const { return bestFitness; }

private:
    int dimensions;
    int hms;
    double hmcr;
    double par;
    double bw;
    int maxIter;
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

    /**
     * Initializes the Harmony Memory with random solutions and their fitness values.
     */
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
 * @param executionType The execution type (e.g., MPI).
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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 8)
    {
        if (rank == 0)
        {
            std::cerr << "Usage: " << argv[0] << " <hms> <hmcr> <par> <bw> <maxIter> <dimensions> <seed>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int hms = std::stoi(argv[1]);
    double hmcr = std::stod(argv[2]);
    double par = std::stod(argv[3]);
    double bw = std::stod(argv[4]);
    int maxIter = std::stoi(argv[5]);
    int dimensions = std::stoi(argv[6]);
    unsigned int seed = std::stoul(argv[7]);

    try
    {
        RosenbrockFunction rosenbrock;
        Solution lowerBounds(dimensions, -5.0);
        Solution upperBounds(dimensions, 5.0);

        MPIHarmonySearch hs(dimensions, hms, hmcr, par, bw, maxIter / size, rosenbrock, lowerBounds, upperBounds, seed);
        Solution bestSolution = hs.optimize();

        if (rank == 0)
        {
            std::cout << "\n==================== Run Start ====================\n";
            std::cout << "Cores: " << size << "\n";
            std::cout << "Dimensions: " << dimensions << "\n";
            std::cout << "HMS: " << hms << ", HMCR: " << hmcr << ", PAR: " << par << ", BW: " << bw << "\n";
            std::cout << "Max Iterations: " << maxIter << ", Seed: " << seed << "\n";
            std::cout << "Execution Time: " << hs.getExecutionTime() << " seconds\n";
            std::cout << "Best fitness: " << hs.getBestFitness() << std::endl;
            std::cout << "==================== Run End ======================" << std::endl;

            writeResultsToCSV("harmony_search_results.csv", dimensions, hms, hmcr, par, bw, maxIter, hs.getExecutionTime(), 
                            MPI::COMM_WORLD.Get_size(), seed, hs.getBestFitness(), "MPI");
        }
    }
    catch (const std::invalid_argument& e) 
    {
        if (rank == 0)
        {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    }

    MPI_Finalize();
    return 0;
}
