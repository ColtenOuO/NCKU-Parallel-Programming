#include <iostream>
#include <string.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <fstream>
#include "mpi.h"

using namespace std;
char message[100] = {};
short w[50000][50000];

struct MinLoc {
    int value;
    int index;
};

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::string file_name;
    if (world_rank == 0) {
        std::cin >> file_name;
        strcpy(message, file_name.c_str());
        message[sizeof(message) - 1] = '\0';
    }

    MPI_Bcast(message, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    std::ifstream input_file(message);
    int n;
    input_file >> n;
    int u;
    int cnt = 0;

    while (input_file >> u) {
        int v;
        input_file >> v;
        int weight;
        input_file >> weight;
        w[u][v] = weight;

        cnt++;
    }

    input_file.close();
    int block = n / world_size;
    int ll = world_rank * block, rr = (world_rank + 1) * block - 1;
    if (rr > n - 1) rr = n - 1;
    vector<int> dis(rr - ll, (int)1e9), ok(n, 0);

    if (world_rank == 0) dis[0] = 0;
    ok[0] = 0;
    int v = 0;
    for (int i = 0; i < n - 1; i++) {
        MPI_Allreduce(MPI_IN_PLACE, dis.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        MinLoc local_min = {(int)1e9, -1};
        for (int i = ll; i <= rr; i++) {
            if (!ok[i] && dis[i - ll] < local_min.value) {
                local_min.value = dis[i - ll];
                local_min.index = i;
            }
        }

        MinLoc global_min = {(int)1e9, -1};
        MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        
        v = global_min.index;
        int dis_value = global_min.value;

        for (int i = ll; i <= rr; i++) {
            if (w[v][i] != 0 && w[v][i] + dis_value < dis[i - ll]) {
                dis[i - ll] = dis_value + w[v][i];
            }
        }

        ok[v] = 1;
    }

    vector<int> final_dis;
    vector<int> all_sz;

    for (int i = 0; i < world_size; i++) {
        int block = n / world_size;
        int ll = i * block, rr = (i + 1) * block - 1;
        if (rr > n - 1) rr = n - 1;
        all_sz.push_back(rr - ll + 1);
    }

    int dis_size = dis.size();
    if (world_rank != 0) {
        MPI_Send(&dis_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        MPI_Send(dis.data(), dis_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        final_dis.insert(final_dis.end(), dis.begin(), dis.end());
        for (int i = 1; i < world_size; i++) {
            int sz;
            MPI_Status status;
            MPI_Recv(&sz, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            vector<int> recv(sz);
            MPI_Recv(recv.data(), sz, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            final_dis.insert(final_dis.end(), recv.begin(), recv.end());
        }
    }

    if (world_rank == 0) {
        for (size_t i = 0; i < final_dis.size(); i++) {
            cout << final_dis[i] << " ";
        }
        cout << endl;
    }

    MPI_Finalize();
    return 0;
}
