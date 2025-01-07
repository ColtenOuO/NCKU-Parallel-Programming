#include <iostream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <pthread.h>
#include <random>
#include <limits>
#include <ctime>

int NUM_CITIES = 20;
const double ALPHA = 1.0; // Influence of pheromone
const double BETA = 5.0;  // Influence of distance
const double RHO = 0.5;  // Pheromone evaporation rate
const double Q = 100.0;  // Pheromone deposit factor
int NUM_ITERATIONS = 1000;
int NUM_ANTS = 10;

std::vector<std::vector<int>> distance_matrix(NUM_CITIES, std::vector<int>(NUM_CITIES));
std::vector<std::vector<double>> pheromone(NUM_CITIES, std::vector<double>(NUM_CITIES, 1.0));

std::vector<int> best_tour;
double best_tour_length = std::numeric_limits<double>::max();

pthread_mutex_t mutex;

void load_distance_matrix(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(1);
    }

    infile >> NUM_CITIES >> NUM_ANTS >> NUM_ITERATIONS;

    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            infile >> distance_matrix[i][j];
        }
    }
    infile.close();
}

double calculate_tour_length(const std::vector<int>& tour) {
    double length = 0.0;
    for (size_t i = 0; i < tour.size() - 1; ++i) {
        length += distance_matrix[tour[i]][tour[i + 1]];
    }
    length += distance_matrix[tour.back()][tour[0]];
    return length;
}

std::vector<int> construct_tour(std::mt19937& rng) {
    std::vector<int> tour;
    std::vector<bool> visited(NUM_CITIES, false);

    std::uniform_int_distribution<int> start_dist(0, NUM_CITIES - 1);
    int current_city = start_dist(rng);
    tour.push_back(current_city);
    visited[current_city] = true;

    for (int step = 1; step < NUM_CITIES; ++step) {
        std::vector<double> probabilities(NUM_CITIES, 0.0);
        double total_prob = 0.0;

        for (int next_city = 0; next_city < NUM_CITIES; ++next_city) {
            if (!visited[next_city]) {
                probabilities[next_city] = std::pow(pheromone[current_city][next_city], ALPHA) *
                                           std::pow(1.0 / distance_matrix[current_city][next_city], BETA);
                total_prob += probabilities[next_city];
            }
        }

        std::uniform_real_distribution<double> prob_dist(0.0, total_prob);
        double random_value = prob_dist(rng);

        for (int next_city = 0; next_city < NUM_CITIES; ++next_city) {
            if (!visited[next_city]) {
                random_value -= probabilities[next_city];
                if (random_value <= 0.0) {
                    current_city = next_city;
                    tour.push_back(current_city);
                    visited[current_city] = true;
                    break;
                }
            }
        }
    }

    return tour;
}

void* ant_worker(void* arg) {
    int rank = *(int*)arg;
    std::mt19937 rng(rank + time(nullptr));

    for (int iteration = 0; iteration < NUM_ITERATIONS; ++iteration) {
        std::vector<std::vector<int>> all_tours(NUM_ANTS);
        std::vector<double> all_lengths(NUM_ANTS, 0.0);

        for (int ant = 0; ant < NUM_ANTS; ++ant) {
            all_tours[ant] = construct_tour(rng);
            all_lengths[ant] = calculate_tour_length(all_tours[ant]);

            pthread_mutex_lock(&mutex);
            if (all_lengths[ant] < best_tour_length) {
                best_tour_length = all_lengths[ant];
                best_tour = all_tours[ant];
            }
            pthread_mutex_unlock(&mutex);
        }

        // Update pheromones
        pthread_mutex_lock(&mutex);
        for (int i = 0; i < NUM_CITIES; ++i) {
            for (int j = 0; j < NUM_CITIES; ++j) {
                pheromone[i][j] *= (1.0 - RHO);
            }
        }

        for (int ant = 0; ant < NUM_ANTS; ++ant) {
            double contribution = Q / all_lengths[ant];
            for (size_t i = 0; i < all_tours[ant].size() - 1; ++i) {
                int from = all_tours[ant][i];
                int to = all_tours[ant][i + 1];
                pheromone[from][to] += contribution;
                pheromone[to][from] += contribution;
            }
        }
        pthread_mutex_unlock(&mutex);
    }

    return nullptr;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pthread_mutex_init(&mutex, nullptr);

    // Broadcast distance matrix to all processes
    for (int i = 0; i < NUM_CITIES; ++i) {
        MPI_Bcast(distance_matrix[i].data(), NUM_CITIES, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    }

    int m = 4; // Number of threads
    pthread_t threads[m];
    int thread_ids[m];

    for (int i = 0; i < m; ++i) {
        thread_ids[i] = rank * m + i;
        pthread_create(&threads[i], nullptr, ant_worker, &thread_ids[i]);
    }

    for (int i = 0; i < m; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Gather best results
    struct {
        double length;
        int rank;
    } local_best = {best_tour_length, rank}, global_best;

    MPI_Reduce(&local_best, &global_best, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Best tour length: " << global_best.length << " by rank " << global_best.rank << "\n";
        std::cout << "Best tour: ";
        for (int city : best_tour) {
            std::cout << city << " ";
        }
        std::cout << std::endl;
    }

    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return 0;
}
