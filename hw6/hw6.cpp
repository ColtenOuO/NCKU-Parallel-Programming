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

int NUM_CITIES;
const double ALPHA = 1.0; // Influence of pheromone
const double BETA = 5.0;  // Influence of distance
const double RHO = 0.5;  // Pheromone evaporation rate
const double Q = 100.0;  // Pheromone deposit factor
int NUM_ITERATIONS;
int NUM_ANTS;
std::vector<std::vector<int>> distance_matrix(NUM_CITIES, std::vector<int>(NUM_CITIES));
std::vector<std::vector<double>> pheromone(NUM_CITIES, std::vector<double>(NUM_CITIES, 1.0));

std::vector<int> best_route;
int best_route_length = 1e9;

pthread_mutex_t mutex;

void load_distance_matrix(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile) {
        std::cout << "file open error\n";
        exit(1);
    }

    infile >> NUM_CITIES >> NUM_ANTS >> NUM_ITERATIONS;
    distance_matrix.resize(NUM_CITIES, std::vector<int>(NUM_CITIES));
    pheromone.resize(NUM_CITIES, std::vector<double>(NUM_CITIES, 1.0));

    for (int i = 0; i < NUM_CITIES; ++i) {
        for (int j = 0; j < NUM_CITIES; ++j) {
            infile >> distance_matrix[i][j];
        }
    }
    infile.close();
    return;
}
int get_route_length(const std::vector<int>& route) {
    int total = 0;
    for(size_t i=0;i<(int)route.size()-1;i++) {
        total += distance_matrix[route[i]][route[i+1]];
    }
    total += distance_matrix[route[(int)route.size()-1]][0];
    return total;
}
std::vector<int> create_route(std::mt19937& rng) {
    std::vector<int> route;
    std::vector<bool> visited(NUM_CITIES, false);
    std::uniform_int_distribution<int> start_dist(0, NUM_CITIES - 1); // random [0, NUM_CITIES-1]
    int current_city = start_dist(rng);
    route.push_back(current_city);
    visited[current_city] = true;
    for(int step = 0; step < NUM_CITIES; step++) {
        std::vector<double> probabilities(NUM_CITIES, 0.0);
        double total_prob = 0.0;

        for (int next_city = 0; next_city < NUM_CITIES; next_city++) {
            if ( !visited[next_city] ) {
                probabilities[next_city] = std::pow(pheromone[current_city][next_city], ALPHA) *
                                           std::pow(1.0 / distance_matrix[current_city][next_city], BETA);
                total_prob += probabilities[next_city];
            }
        }

        std::uniform_real_distribution<double> prob_dist(0.0, total_prob);
        double random_value = prob_dist(rng);
        for(int next_city = 0; next_city < NUM_CITIES; next_city++) {
            if( !visited[next_city] ) {
                random_value -= probabilities[next_city];
                if( random_value <= 0.0 ) {
                    current_city = next_city;
                    route.push_back(current_city);
                    visited[current_city] = true;
                    break;
                }
            }
        }
    }

    return route;
}
void *ant_worker(void* arg) {
    int rank = *(int*)arg;
    std::mt19937 rng(rank + time(nullptr));
    std::vector<std::vector<int>> all_route(NUM_ANTS);
    std::vector<int> all_lengths(NUM_ANTS, 0);
    for(int iteration = 0; iteration < NUM_ITERATIONS; iteration++) {
        for(int ant = 0; ant < NUM_ANTS; ant++) {
            all_route[ant] = create_route(rng);
            all_lengths[ant] = get_route_length(all_route[ant]);
            
            pthread_mutex_lock(&mutex);
            if( all_lengths[ant] < best_route_length ) {
                best_route_length = all_lengths[ant];
                best_route = all_route[ant];
            }
            pthread_mutex_unlock(&mutex);
        }
    }

    // Update pheromones
    pthread_mutex_lock(&mutex);
    for (int i = 0; i < NUM_CITIES; i++) {
        for (int j = 0; j < NUM_CITIES; j++) {
            pheromone[i][j] *= (1.0 - RHO);
        }
    }

    for (int ant = 0; ant < NUM_ANTS; ant++) {
            double contribution = Q / all_lengths[ant];
            for (int i = 0; i < (int)all_route[ant].size() - 1; i++) {
                int from = all_route[ant][i];
                int to = all_route[ant][i + 1];
                pheromone[from][to] += contribution;
                pheromone[to][from] += contribution;
            }
        }
    pthread_mutex_unlock(&mutex);

    return nullptr;
}
int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    pthread_mutex_init(&mutex, nullptr);

    if (rank == 0) {
        std::string file_name;
        std::cin >> file_name;
        load_distance_matrix(file_name);
    }

    MPI_Bcast(&NUM_CITIES, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_ANTS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&NUM_ITERATIONS, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if( rank != 0 ) {
        distance_matrix.resize(NUM_CITIES, std::vector<int>(NUM_CITIES));
        pheromone.resize(NUM_CITIES, std::vector<double>(NUM_CITIES, 1.0));
    }

    for (int i = 0; i < NUM_CITIES; i++) {
        MPI_Bcast(distance_matrix[i].data(), NUM_CITIES, MPI_INT, 0, MPI_COMM_WORLD);
    }

    int m = 4; // Number of threads
    pthread_t threads[m];
    int thread_ids[m];

    for (int i = 0; i < m; i++) {
        thread_ids[i] = rank * m + i;
        pthread_create(&threads[i], nullptr, ant_worker, &thread_ids[i]);
    }

    for (int i = 0; i < m; i++) {
        pthread_join(threads[i], nullptr);
    }

    struct {
        int length;
        int rank;
    } local_best = {best_route_length, rank}, global_best;

    MPI_Reduce(&local_best, &global_best, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << "Best tour length: " << global_best.length << " by rank " << global_best.rank << "\n";
        std::cout << "Best tour: ";
        for (int city : best_route) {
            std::cout << city << " ";
        }
        std::cout << std::endl;
    }

    pthread_mutex_destroy(&mutex);
    MPI_Finalize();
    return 0;
}