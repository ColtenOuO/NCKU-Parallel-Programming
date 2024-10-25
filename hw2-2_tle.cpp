#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

vector<pair<int,int>> e[50001];
char message[100] = {};
int main(int argc, char *argv[]) {
   std::ios_base::sync_with_stdio(false);
   std::cin.tie(0);
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::string file_name;
    if( world_rank == 0 ) {
        std::cin >> file_name;
        strcpy(message, file_name.c_str());
        message[sizeof(message) - 1] = '\0';
    }

    MPI_Bcast(message, 100, MPI_CHAR, 0, MPI_COMM_WORLD);
    std::ifstream input_file(message);
    int n;
    input_file >> n;
    int u;

    int block = 50000 / world_size;
    int cnt = 0;
    while( input_file >> u ) {
        int v;
        input_file >> v;
        int w;
        input_file >> w;
        if( cnt < ( world_rank + 1 ) * block && cnt >= world_rank * block ) {
            e[u].push_back({v,w});
            e[v].push_back({u,w});
        }

        cnt++;
    }
    input_file.close();
    vector<int> dis(n,(int)1e9),ok(n,0);
    dis[0] = 0;
    ok[0] = 0;
    int v = 0;
    for(int i = 0; i < n - 1; i++) {
        MPI_Allreduce(MPI_IN_PLACE, dis.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        
        int mn = 1e9, pos = -1;
        for (int i = 0; i < n; i++) {
            if (ok[i] == 0 && mn > dis[i]) {
                mn = dis[i];
                pos = i;
            }
        }
        ok[pos] = 1;
        v = pos;
        
        for (auto j : e[v]) {
            dis[j.first] = min(dis[j.first], dis[v] + j.second); 
        }
    }

    if( world_rank == 0 ) {
        string ans = "";
        for(int i=0;i<n;i++) {
            ans += to_string(dis[i]);
            ans += " ";
        }

        cout << ans;
    }

    MPI_Finalize();
    return 0;
}
