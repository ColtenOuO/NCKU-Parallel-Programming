#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <mpi.h>
#include <algorithm>

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<int> test[40];
    long long cost[40];
    int n, m;

    if (world_rank == 0 ) {
        std::string file_name;
        std::cin >> file_name;
        std::ifstream input_file(file_name);
        input_file >> n >> m;

        for (int i = 0; i < m; i++) {
            int cnt, c;
            input_file >> cnt >> c;
            cost[i] = c;
            for (int j = 0; j < cnt; j++) {
                int id;
                input_file >> id;
                test[i].emplace_back(id);
            }
        }
        input_file.close();
    }

    // Broadcast n and m to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(cost, m, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    int start_index, end_index;
    bool check = false;
    // Divide data between two processes
    if (world_rank == 0) {
        // Send data to process 1
        for (int i = m / 2; i < m; i++) {
            int size = test[i].size();
            MPI_Send(&size, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
            if (size > 0) {
                MPI_Send(test[i].data(), size, MPI_INT, 1, 0, MPI_COMM_WORLD);       
            }
        }
        start_index = 0;
        end_index = m / 2;
    } else if (world_rank == 1) {
        // Receive data from process 0
        check = true;
        for (int i = m / 2; i < m; i++) {
            int size;
            MPI_Recv(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            test[i].resize(size); // must have, otherwise will null ptr
            if (size > 0) {
                MPI_Recv(test[i].data(), size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        start_index = m / 2;
        end_index = m;
    } else {
        // For other processes, no data in initial division
        start_index = end_index = 0;
    }

    // Prepare local data
    std::vector<std::vector<int>> v;
    std::vector<long long> v_bit;
    std::vector<std::pair<long long,int>> result;
    std::vector<int> cost2;

    for (int i = start_index; i < end_index; i++) {
        v.push_back(test[i]);
    }

    for (int i = 0; i < v.size(); i++) {
        long long num = 0;
        for (auto j : v[i]) {
            num |= (1LL << j);
        }
        v_bit.push_back(num);
        cost2.push_back(cost[i + start_index]);
    }

    int subset_size = v.size();
    for (int i = 0; i < (1 << subset_size); i++) {
        long long bit = 0;
        int total_cost = 0;
        for (int j = 0; j < subset_size; j++) {
            if (i & (1 << j)) {
                bit |= v_bit[j];
                total_cost += cost2[j];
            }
        }
        result.emplace_back(bit, total_cost);
    }

    if (world_rank != 0) {
        int result_size = result.size();
        MPI_Send(&result_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        for (const auto& r : result) {
            MPI_Send(&r.first, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&r.second, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    std::vector<std::pair<long long, int>> final_result2;
    long long total_ans = 0;
    if (world_rank == 0) {
        for (int i = 1; i < 2; i++) {
            int result_size;
            MPI_Recv(&result_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int j = 0; j < result_size; j++) {
                long long bit;
                int total_cost;
                MPI_Recv(&bit, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&total_cost, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                final_result2.emplace_back(bit, total_cost);
            }
        }

        int size1 = result.size();
        int size2 = final_result2.size();
        std::vector<long long> result1_bits(size1);
        std::vector<int> result1_costs(size1);

        for (int i = 0; i < size1; ++i) {
            result1_bits[i] = result[i].first;
            result1_costs[i] = result[i].second;
        }

        std::vector<long long> result2_bits(size2);
        std::vector<int> result2_costs(size2);

        for (int i = 0; i < size2; ++i) {
            result2_bits[i] = final_result2[i].first;
            result2_costs[i] = final_result2[i].second;
        }

        MPI_Bcast(&size1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&size2, 1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(result1_bits.data(), size1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(result1_costs.data(), size1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(result2_bits.data(), size2, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(result2_costs.data(), size2, MPI_INT, 0, MPI_COMM_WORLD);
        long long ans = 0;
        int block_size = result1_bits.size() / 8;
        int index_begin = world_rank * block_size, index_end = std::min(index_begin + block_size, (int)result1_bits.size());
        if( block_size == 0 ) index_end = (int)result1_bits.size();
        for(int i=index_begin;i<index_end;i++) {
            for(int j=0;j<result2_bits.size();j++) {
                long long final_bit = ( result2_bits[j] | result1_bits[i] );
                if( final_bit == ( 1LL << ( n + 1 ) ) - 2 ) ans++;
            }
        }

        MPI_Reduce(&ans, &total_ans, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    } else {
        int size1, size2;
        MPI_Bcast(&size1, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&size2, 1, MPI_INT, 0, MPI_COMM_WORLD);
        std::vector<long long> result1_bits(size1);
        std::vector<int> result1_costs(size1);

        std::vector<long long> result2_bits(size2);
        std::vector<int> result2_costs(size2);

        MPI_Bcast(result1_bits.data(), size1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(result1_costs.data(), size1, MPI_INT, 0, MPI_COMM_WORLD);

        MPI_Bcast(result2_bits.data(), size2, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(result2_costs.data(), size2, MPI_INT, 0, MPI_COMM_WORLD);
        long long ans = 0;
        int block_size = result1_bits.size() / 8;
        int index_begin = world_rank * block_size, index_end = std::min(index_begin + block_size, (int)result1_bits.size());
        if( block_size == 0 ) index_begin = index_end;
        for(int i=index_begin;i<index_end;i++) {
            for(int j=0;j<result2_bits.size();j++) {
                long long final_bit = ( result2_bits[j] | result1_bits[i] );
                if( final_bit == ( 1LL << ( n + 1 ) ) - 2 ) ans++;
            }
        }
        MPI_Reduce(&ans, &total_ans, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (world_rank == 0) {
        std::cout << total_ans;
    }

    MPI_Finalize();
    return 0;
}
