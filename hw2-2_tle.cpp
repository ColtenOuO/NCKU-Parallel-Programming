#pragma GCC optimize(3)

#pragma GCC target("avx")

#pragma GCC optimize("Ofast")

#pragma GCC optimize("inline")

#pragma GCC optimize("-fgcse")

#pragma GCC optimize("-fgcse-lm")

#pragma GCC optimize("-fipa-sra")

#pragma GCC optimize("-ftree-pre")

#pragma GCC optimize("-ftree-vrp")

#pragma GCC optimize("-fpeephole2")

#pragma GCC optimize("-ffast-math")

#pragma GCC optimize("-fsched-spec")

#pragma GCC optimize("unroll-loops")

#pragma GCC optimize("-falign-jumps")

#pragma GCC optimize("-falign-loops")

#pragma GCC optimize("-falign-labels")

#pragma GCC optimize("-fdevirtualize")

#pragma GCC optimize("-fcaller-saves")

#pragma GCC optimize("-fcrossjumping")

#pragma GCC optimize("-fthread-jumps")

#pragma GCC optimize("-funroll-loops")

#pragma GCC optimize("-fwhole-program")

#pragma GCC optimize("-freorder-blocks")

#pragma GCC optimize("-fschedule-insns")

#pragma GCC optimize("inline-functions")

#pragma GCC optimize("-ftree-tail-merge")

#pragma GCC optimize("-fschedule-insns2")

#pragma GCC optimize("-fstrict-aliasing")

#pragma GCC optimize("-fstrict-overflow")

#pragma GCC optimize("-falign-functions")

#pragma GCC optimize("-fcse-skip-blocks")

#pragma GCC optimize("-fcse-follow-jumps")

#pragma GCC optimize("-fsched-interblock")

#pragma GCC optimize("-fpartial-inlining")

#pragma GCC optimize("no-stack-protector")

#pragma GCC optimize("-freorder-functions")

#pragma GCC optimize("-findirect-inlining")

#pragma GCC optimize("-fhoist-adjacent-loads")

#pragma GCC optimize("-frerun-cse-after-loop")

#pragma GCC optimize("inline-small-functions")

#pragma GCC optimize("-finline-small-functions")

#pragma GCC optimize("-ftree-switch-conversion")

#pragma GCC optimize("-foptimize-sibling-calls")

#pragma GCC optimize("-fexpensive-optimizations")

#pragma GCC optimize("-funsafe-loop-optimizations")

#pragma GCC optimize("inline-functions-called-once")

#pragma GCC optimize("-fdelete-null-pointer-checks")

#pragma GCC optimize(2)

#include <iostream>

#include <string.h>

#include <vector>

#include <utility>

#include <algorithm>

#include <fstream>

#include "mpi.h"



using namespace std;



char message[100] = {};

vector<pair<int, int>> e[500001];



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

    if (!input_file.is_open()) {

        if (world_rank == 0) {

            cerr << "Error: Could not open file " << message << endl;

        }

        MPI_Finalize();

        return 1;

    }



    int n;

    input_file >> n;



    int u, v, weight;

    while (input_file >> u >> v >> weight) {

        int block = n / world_size;

        if( block == 0 ) block = 1;

        int ll = world_rank * block;

        int rr = (world_rank + 1) * block - 1;



        if (world_rank == world_size - 1 || rr >= n ) {

            rr = n - 1;

        }

        if( ll >= n ) ll = n - 1;

	//cout << world_rank << " " << ll << " " << rr << "\n";

        if (ll <= v && v <= rr) {

            e[u].push_back(make_pair(v, weight));

        }

    }

    input_file.close();



    int block = n / world_size;

    if( block == 0 ) block = 1;

    int ll = world_rank * block;

    int rr = (world_rank + 1) * block - 1;

    if (world_rank == world_size - 1 || rr >= n ) {

        rr = n - 1;

    }

    if( ll >= n ) ll = n - 1;



    vector<int> dis(n, (int)1e9);

    vector<int> ok(n, 0);



    if (world_rank == 0) dis[0] = 0;

    for (int i = 0; i < n - 1; i++) {

        MinLoc local_min = {(int)1e9, -1};

        for (int i = ll; i <= rr; i++) {

            if (dis[i] < local_min.value && ok[i] == 0) {

                local_min.value = dis[i];

                local_min.index = i;

            }

        }

        



        MinLoc global_min = {(int)1e9, -1};

        MPI_Allreduce(&local_min, &global_min, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        v = global_min.index;

       // cout << "i = " << i << " v= " << v << "\n";

        int dis_value = global_min.value;



        if (v != -1 && dis_value != (int)1e9) {  // Only proceed if v is valid

            for (auto i : e[v]) {

                if (dis_value + i.second < dis[i.first]) dis[i.first] = dis_value + i.second;

            }



            ok[global_min.index] = 1;

        }

    }



    vector<int> global_dis(n, (int)1e9);

    MPI_Allreduce(dis.data(), global_dis.data(), n, MPI_INT, MPI_MIN, MPI_COMM_WORLD);



    if (world_rank == 0) {

    	string ans = "";

        for (size_t i = 0; i < global_dis.size(); i++) {

            ans += to_string(global_dis[i]);

            ans += " ";

        }

        cout << ans;

    }



    MPI_Finalize();

    return 0;

}

