// Parallel Harmony Search Algorithm Implementation in C++ with MPI

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
#include <mpi.h>

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
                  const ObjectiveFunctionBase& objFunc, const Solution& lowerBounds, const Solution& upperBounds, 
                  unsigned int seed, int rank, int numProcs)
        : dimensions(dimensions), hms(hms), hmcr(hmcr), par(par), bw(bw), maxIter(maxIter),
          objectiveFunction(objFunc), lowerBounds(lowerBounds), upperBounds(upperBounds), rng(seed),
          rank(rank), numProcs(numProcs), BATCH_SIZE(1000 * numProcs)
    { 
        if (hms <= 0) throw std::invalid_argument("Harmony memory size (hms) must be greater than 0.");
        if (hmcr < 0.0 || hmcr > 1.0) throw std::invalid_argument("HMCR must be in [0, 1].");
        if (par < 0.0 || par > 1.0) throw std::invalid_argument("PAR must be in [0, 1].");
        if (bw <= 0.0) throw std::invalid_argument("Bandwidth (bw) must be greater than 0.");
    }

    /**
     * Optimizes the objective function using Harmony Search.
     * @return The best solution found.
     */
    Solution optimize()
    {
        auto start = std::chrono::high_resolution_clock::now();
        initializeHarmonyMemory();

        if (numProcs == 1) 
        {
            for (int iter = 0; iter < maxIter; ++iter) 
            {
                Solution newHarmony = generateNewHarmony();
                double newFitness = objectiveFunction.evaluate(newHarmony);
                if (newFitness < worstFitness) 
                {
                    replaceWorstHarmony(newHarmony, newFitness);
                    recomputeWorstAndBest();
                }
            }
        } 
        else 
        {
            int fullBatches = maxIter / BATCH_SIZE;
            int remainingIterations = maxIter % BATCH_SIZE;

            for (int batch = 0; batch < fullBatches; ++batch) 
            {
                std::vector<std::pair<double, Solution>> localBest;

                // Generate batch with early filtering
                for (int i = 0; i < BATCH_SIZE; ++i) 
                {
                    Solution newHarmony = generateNewHarmony();
                    double newFitness = objectiveFunction.evaluate(newHarmony);
                    if (newFitness < worstFitness * 0.95) 
                    {
                        localBest.emplace_back(newFitness, newHarmony);
                    }
                }

                // Prepare send buffer
                std::vector<double> sendBuffer;
                for (const auto& elem : localBest) 
                {
                    sendBuffer.push_back(elem.first);
                    const Solution& harmony = elem.second;
                    sendBuffer.insert(sendBuffer.end(), harmony.begin(), harmony.end());
                }

                // Gather data sizes
                int sendCount = localBest.size();
                std::vector<int> recvCounts(numProcs);
                MPI_Allgather(&sendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

                // Convert counts to doubles (each element is dimensions+1 doubles)
                std::vector<int> recvCountsDoubles(numProcs);
                for (int i = 0; i < numProcs; ++i) 
                {
                    recvCountsDoubles[i] = recvCounts[i] * (dimensions + 1);
                }

                // Calculate displacements and total elements
                std::vector<int> displs(numProcs);
                int totalElements = 0;
                for (int i = 0; i < numProcs; ++i) 
                {
                    displs[i] = totalElements;
                    totalElements += recvCountsDoubles[i];
                }

                // Gather all data
                std::vector<double> allData(totalElements);
                MPI_Allgatherv(sendBuffer.data(), sendCount * (dimensions + 1), MPI_DOUBLE,
                            allData.data(), recvCountsDoubles.data(), displs.data(), 
                            MPI_DOUBLE, MPI_COMM_WORLD);

                processReceivedHarmonies(allData, totalElements);
            }

            // Handle remaining iterations
            if (remainingIterations > 0) 
            {
                std::vector<std::pair<double, Solution>> localBest;
                for (int i = 0; i < remainingIterations; ++i) 
                {
                    Solution newHarmony = generateNewHarmony();
                    double newFitness = objectiveFunction.evaluate(newHarmony);
                    if (newFitness < worstFitness * 0.95) 
                    {
                        localBest.emplace_back(newFitness, newHarmony);
                    }
                }

                // Prepare send buffer
                std::vector<double> sendBuffer;
                for (const auto& elem : localBest) 
                {
                    sendBuffer.push_back(elem.first);
                    const Solution& harmony = elem.second;
                    sendBuffer.insert(sendBuffer.end(), harmony.begin(), harmony.end());
                }

                // Gather data sizes
                int sendCount = localBest.size();
                std::vector<int> recvCounts(numProcs);
                MPI_Allgather(&sendCount, 1, MPI_INT, recvCounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

                // Convert counts to doubles
                std::vector<int> recvCountsDoubles(numProcs);
                for (int i = 0; i < numProcs; ++i) 
                {
                    recvCountsDoubles[i] = recvCounts[i] * (dimensions + 1);
                }

                // Calculate displacements and total elements
                std::vector<int> displs(numProcs);
                int totalElements = 0;
                for (int i = 0; i < numProcs; ++i) 
                {
                    displs[i] = totalElements;
                    totalElements += recvCountsDoubles[i];
                }

                // Gather all data
                std::vector<double> allData(totalElements);
                MPI_Allgatherv(sendBuffer.data(), sendCount * (dimensions + 1), MPI_DOUBLE,
                            allData.data(), recvCountsDoubles.data(), displs.data(), 
                            MPI_DOUBLE, MPI_COMM_WORLD);

                processReceivedHarmonies(allData, totalElements);
            }
        }

        if (rank == 0) 
        {
            auto end = std::chrono::high_resolution_clock::now();
            executionTime = std::chrono::duration<double>(end - start).count();
        }

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

    int rank;
    int numProcs;
    const int BATCH_SIZE;

    /**
     * Initializes the Harmony Memory with random solutions and their fitness values.
     */
    void initializeHarmonyMemory()
    {
        // Calculate work distribution
        int hms_local = hms / numProcs;
        int remainder = hms % numProcs;
        std::vector<int> counts(numProcs, hms_local);
        std::vector<int> displs(numProcs, 0);

        // Handle remainder on root
        if (rank == 0) {
            counts[0] += remainder;
        }

        // Generate local solutions
        std::vector<Solution> localHM(counts[rank]);
        std::vector<double> localFitness(counts[rank]);
        
        for (int i = 0; i < counts[rank]; ++i) {
            localHM[i] = randomSolution();
            localFitness[i] = objectiveFunction.evaluate(localHM[i]);
        }

        // Prepare buffers for gathering
        std::vector<double> localHMBuffer;
        for (const auto& sol : localHM) {
            localHMBuffer.insert(localHMBuffer.end(), sol.begin(), sol.end());
        }

        // Calculate displacements and counts for MPI
        std::vector<int> hm_counts(numProcs);
        std::vector<int> hm_displs(numProcs);
        std::vector<int> fit_counts(numProcs);
        std::vector<int> fit_displs(numProcs);

        if (rank == 0) {
            int hm_offset = 0;
            int fit_offset = 0;
            for (int i = 0; i < numProcs; ++i) {
                hm_counts[i] = counts[i] * dimensions;
                hm_displs[i] = hm_offset;
                hm_offset += hm_counts[i];

                fit_counts[i] = counts[i];
                fit_displs[i] = fit_offset;
                fit_offset += fit_counts[i];
            }
        }

        // Gather harmony memory
        std::vector<double> hmBuffer;
        std::vector<double> fitnessBuffer;
        
        if (rank == 0) {
            hmBuffer.resize(hms * dimensions);
            fitnessBuffer.resize(hms);
        }

        MPI_Gatherv(localHMBuffer.data(), counts[rank] * dimensions, MPI_DOUBLE,
                hmBuffer.data(), hm_counts.data(), hm_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

        MPI_Gatherv(localFitness.data(), counts[rank], MPI_DOUBLE,
                fitnessBuffer.data(), fit_counts.data(), fit_displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

        // Broadcast full harmony memory
        if (rank == 0) {
            MPI_Bcast(hmBuffer.data(), hms * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(fitnessBuffer.data(), hms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        } else {
            hmBuffer.resize(hms * dimensions);
            fitnessBuffer.resize(hms);
            MPI_Bcast(hmBuffer.data(), hms * dimensions, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(fitnessBuffer.data(), hms, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }

        // Reconstruct harmony memory
        harmonyMemory.resize(hms);
        for (int i = 0; i < hms; ++i) {
            harmonyMemory[i] = Solution(
                hmBuffer.begin() + i * dimensions,
                hmBuffer.begin() + (i + 1) * dimensions
            );
        }
        fitness = fitnessBuffer;

        recomputeWorstAndBest();
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

    void recomputeWorstAndBest()
    {
        worstIndex = std::distance(fitness.begin(), std::max_element(fitness.begin(), fitness.end()));
        worstFitness = fitness[worstIndex];
        auto bestIt = std::min_element(fitness.begin(), fitness.end());
        bestFitness = *bestIt;
        bestSolution = harmonyMemory[std::distance(fitness.begin(), bestIt)];
    }

    void processReceivedHarmonies(const std::vector<double>& allData, size_t totalElements)
    {
        std::vector<std::pair<double, Solution>> candidates;
        size_t pos = 0;
        
        while (pos < totalElements) 
        {
            double fitnessVal = allData[pos++];
            Solution harmony(allData.begin() + pos, allData.begin() + pos + dimensions);
            pos += dimensions;
            candidates.emplace_back(fitnessVal, std::move(harmony));
        }

        std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<double, Solution>& a, const std::pair<double, Solution>& b) 
            {
                return a.first < b.first;
            });

        // Replace worst harmonies in bulk
        int replaceCount = std::min(static_cast<int>(candidates.size()), hms);
        for (int i = 0; i < replaceCount; ++i) 
        {
            int targetIndex = hms - 1 - i;
            if (candidates[i].first < fitness[targetIndex]) 
            {
                harmonyMemory[targetIndex] = candidates[i].second;
                fitness[targetIndex] = candidates[i].first;
            }
        }

        recomputeWorstAndBest();
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

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (argc != 8) {
        if (rank == 0) std::cerr << "Usage: " << argv[0] << " <hms> <hmcr> <par> <bw> <maxIter> <dimensions> <seed>\n";
        MPI_Finalize();
        return 1;
    }

    try {
        int hms = std::stoi(argv[1]);
        double hmcr = std::stod(argv[2]);
        double par = std::stod(argv[3]);
        double bw = std::stod(argv[4]);
        int maxIter = std::stoi(argv[5]);
        int dimensions = std::stoi(argv[6]);
        unsigned int originalSeed = std::stoul(argv[7]);
        unsigned int seed = originalSeed + rank;

        maxIter = (maxIter + numProcs - 1) / numProcs;

        RosenbrockFunction rosenbrock;
        Solution lowerBounds(dimensions, -5.0);
        Solution upperBounds(dimensions, 5.0);

        HarmonySearch hs(dimensions, hms, hmcr, par, bw, maxIter, rosenbrock, lowerBounds, upperBounds, seed, rank, numProcs);
        Solution best = hs.optimize();

        if (rank == 0) {
            std::cout << "\n==================== Run Summary ====================\n"
                      << "Cores: " << numProcs << "\n"
                      << "Dimensions: " << dimensions << "\n"
                      << "HMS: " << hms << ", HMCR: " << hmcr << ", PAR: " << par << ", BW: " << bw << "\n"
                      << "Max Iterations: " << maxIter * numProcs << "\n"
                      << "Execution Time: " << hs.getExecutionTime() << " seconds\n"
                      << "Best fitness: " << hs.getBestFitness() << "\n"
                      << "=====================================================\n";

            writeResultsToCSV("Parallel-Harmony-Search/data/harmony_search_results.csv", dimensions, hms, hmcr, par, bw, maxIter * numProcs, 
                              hs.getExecutionTime(), numProcs, originalSeed, hs.getBestFitness(), "MPI");
        }
    } catch (const std::exception& e) {
        if (rank == 0) std::cerr << "Error: " << e.what() << "\n";
        MPI_Finalize();
        return 1;
    }

    MPI_Finalize();
    return 0;
}