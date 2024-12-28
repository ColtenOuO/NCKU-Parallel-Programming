#include <iostream>
#include <pthread.h>
#include <vector>
#include <fstream>
#include <string>
#include <string.h>
#include <mpi.h>

std::vector<int> sequence;
int n, local_ans = 0;
long long arr1[10001], arr2[10001];
pthread_mutex_t mutex;

struct ThreadArg {
    long long start;
    long long end;
};

void* processSubsequence(void* arg) {
    ThreadArg* range = static_cast<ThreadArg*>(arg);
    long long start = range->start;
    long long end = range->end;
    const unsigned long long ALL_BITS_SET = ~0ULL;
    for (long long i = start; i <= end; i++) {
        for(int j=0;j<n;j++) {
            if( ( __builtin_popcountll(( ~( i ^ arr1[j] ) ) | arr2[j] )) == 64 ) {
                pthread_mutex_lock(&mutex);
                local_ans++;
                pthread_mutex_unlock(&mutex);
                break;
            }
        }

    }

    return nullptr;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int m = 4;
    pthread_t threads[m];
    if (rank == 0) {
        std::cin.tie(0)->sync_with_stdio(false);
    }

    std::string file_name;
    if (rank == 0) {
        std::cin >> file_name;
    }

    char file_name_c[256] = {};
    if (rank == 0) {
        strncpy(file_name_c, file_name.c_str(), 255);
    }
    MPI_Bcast(file_name_c, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    file_name = file_name_c;

    if (rank == 0) {
        std::ifstream input_file(file_name);
        input_file >> n;
        for (int i = 0; i < n; i++) {
            long long num;
            input_file >> num;
            int cnt = 20;
            while( num != 0 ) {
                cnt--;
                if( num % 3 == 0 ) arr1[i] *= 2, arr2[i] *= 2, arr2[i] += 1;
                if( num % 3 == 1 ) arr1[i] *= 2, arr1[i] += 1, arr2[i] *= 2;
                if( num % 3 == 2 ) arr1[i] *= 2, arr2[i] *= 2; 
                num /= 3;
            }
            while(cnt--) arr1[i] *= 2, arr2[i] *= 2, arr2[i] += 1;
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(arr1, 10001, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(arr2, 10001, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    const long long total = ( 1LL << 20 );
    long long range_per_rank = total / size;
    long long rank_start = rank * range_per_rank;
    long long rank_end = (rank == size - 1) ? total - 1 : rank_start + range_per_rank - 1;

    pthread_mutex_init(&mutex, nullptr);

    long long range_per_thread = (rank_end - rank_start + 1) / m;
    for (int i = 0; i < m; i++) {
        long long thread_start = rank_start + i * range_per_thread;
        long long thread_end = (i == m - 1) ? rank_end : thread_start + range_per_thread - 1;

        ThreadArg* arg = new ThreadArg;
        arg -> start = thread_start;
        arg -> end = thread_end;
        pthread_create(&threads[i], nullptr, processSubsequence, arg);
    }

    for (int i = 0; i < m; i++) {
        pthread_join(threads[i], nullptr);
    }

    pthread_mutex_destroy(&mutex);

    int global_ans = 0;
    MPI_Reduce(&local_ans, &global_ans, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << global_ans;
    }

    MPI_Finalize();
    return 0;
}
