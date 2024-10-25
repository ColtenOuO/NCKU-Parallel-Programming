#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

vector<pair<int, int>> e[50001];
char message[100] = {};

struct MinLoc {
    int value;
    int index;
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    if (world_rank == 0) {
        string file_name;
        cin >> file_name;
        strcpy(message, file_name.c_str());
        message[sizeof(message) - 1] = '\0';
    }

    MPI_Bcast(message, 100, MPI_CHAR, 0, MPI_COMM_WORLD);

    ifstream input_file(message);
    int n;
    input_file >> n;
    int u;

    int block = (50000 + world_size - 1) / world_size;
    int cnt = 0;
    while (input_file >> u) {
        int v, w;
        input_file >> v >> w;
        if (cnt >= world_rank * block && cnt < (world_rank + 1) * block) {
            e[u].push_back({v, w});
        }
        cnt++;
    }
    input_file.close();

    vector<int> dis(n, (int)1e9), ok(n, 0);
    dis[0] = 0;
    ok[0] = 1;
    int v = 0;

    for (int i = 0; i < n - 1; i++) {
        MinLoc local_min = {(int)1e9, -1};
        for (auto j : e[v]) {
            dis[j.first] = min(dis[j.first], dis[v] + j.second);
            if (!ok[j.first] && dis[j.first] < local_min.value) {
                local_min.value = dis[j.first];
                local_min.index = j.first;
            }
        }

        int ll = world_rank * (n / world_size);
        int rr = (world_rank + 1) * (n / world_size) - 1;
        if (world_rank == world_size - 1) rr = n - 1;

        for (int j = 0; j < n; j++) {
            if (!ok[j] && dis[j] < local_min.value) {
                local_min.value = dis[j];
                local_min.index = j;
            }
        }
        
        MinLoc global_min = {(int)1e9, -1};
        MPI_Reduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
        MPI_Bcast(&global_min, 1, MPI_2INT, 0, MPI_COMM_WORLD);
        

        if (global_min.index == -1) break;

        ok[global_min.index] = 1;
        v = global_min.index;
        dis[v] = global_min.value;
    }
    

    if (world_rank == 0) {
        string ans;
        for (int i = 0; i < n; i++) {
            ans += to_string(dis[i]) + " ";
        }
        cout << ans;
    }

    MPI_Finalize();
    return 0;
}
