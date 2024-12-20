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
#include <fstream>
#include <vector>
#include <utility>
#include <cmath>
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <unistd.h>
#pragma omp parallel for

int wrapIndex(int idx, int limit) {
    if (idx < 0) return limit - 1;
    if (idx >= limit) return 0;
    return idx;
}

double solve(int n, int m, int i, int j, int offset, std::vector<std::vector<int> > &A, std::vector<std::vector<int> > &K) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum) collapse(2)
    for (int di = -offset; di <= offset; ++di) {
        for (int dj = -offset; dj <= offset; ++dj) {
            int ni = wrapIndex(i + di, m); // wrap row index
            int nj = wrapIndex(j + dj, n); // wrap column index
            sum += K[di + offset][dj + offset] * A[ni][nj];
        }
    }
    return sum;
}

int main(int argc, char *argv[]) {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(0);
    MPI_Init(&argc, &argv);
    
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::vector<std::vector<int> > A;
    std::vector<std::vector<int> > K;

    int t, n, m, D;
    if (world_rank == 0) {
        std::string file_name;
        std::cin >> file_name;
        std::ifstream input_file(file_name);
        input_file >> t >> n >> m;
        A.resize(m, std::vector<int>(n));
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                input_file >> A[i][j];
            }
        }
        input_file >> D;
        K.resize(D, std::vector<int>(D));
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                input_file >> K[i][j];
            }
        } 
        input_file.close();
    }

    std::vector<int> flat_A;
    std::vector<int> flat_K;

    if (world_rank == 0) {
        flat_A.resize(m * n);
        flat_K.resize(D * D);

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                flat_A[i * n + j] = A[i][j];
            }
        }
        for (int i = 0; i < D; ++i) {
            for (int j = 0; j < D; ++j) {
                flat_K[i * D + j] = K[i][j];
            }
        }
    }

    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&D, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&t, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        flat_A.resize(m * n);
        flat_K.resize(D * D);
    }

    MPI_Bcast(flat_A.data(), m * n, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(flat_K.data(), D * D, MPI_INT, 0, MPI_COMM_WORLD);

    A.resize(m, std::vector<int>(n));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A[i][j] = flat_A[i * n + j];
        }
    }

    K.resize(D, std::vector<int>(D));
    for (int i = 0; i < D; ++i) {
        for (int j = 0; j < D; ++j) {
            K[i][j] = flat_K[i * D + j];
        }
    }

    int offset = D / 2;
    std::vector<std::vector<int> > A_next(m, std::vector<int>(n));

    if (world_rank == 0) {
        #pragma omp parallel for
        for (int step = 0; step < t; ++step) {
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;

                    if (offset * 2 < 8) {
                        sum = solve(n, m, i, j, offset, A, K);
                    } else {
                        int ll = -offset, rr = (offset * 2) / 8 - 1;
                        sum = solve(n, m, i, j, offset, A, K);

                        MPI_Request request[8];
                        for (int k = 1; k < 8; ++k) {
                            int sz = offset * 2, block = sz / 8;
                            int l = k * block, r = l + block - 1;
                            if (r > 2 * offset) r = 2 * offset;
                            l -= offset;
                            r -= offset;
                            MPI_Isend(&i, 1, MPI_INT, k, 0, MPI_COMM_WORLD, &request[0]);
                            MPI_Isend(&j, 1, MPI_INT, k, 0, MPI_COMM_WORLD, &request[1]);
                            MPI_Isend(&l, 1, MPI_INT, k, 0, MPI_COMM_WORLD, &request[2]);
                            MPI_Isend(&r, 1, MPI_INT, k, 0, MPI_COMM_WORLD, &request[3]);
                        }

                        for (int k = 1; k < 8; ++k) {
                            double total = 0.0;
                            MPI_Recv(&total, 1, MPI_DOUBLE, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                            sum += total;
                        }
                    }

                    A_next[i][j] = static_cast<int>(floor(sum / (D * D)));
                }
            }

            A = A_next;
        }
    } else if (offset * 2 >= 8) {
        int l, r, i, j;
        MPI_Recv(&i, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&j, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&l, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double sum = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int di = l; di <= r; ++di) {
            for (int dj = l; dj <= r; ++dj) {
                int ni = wrapIndex(i + di, m); // wrap row index
                int nj = wrapIndex(j + dj, n); // wrap column index
                sum += K[di + offset][dj + offset] * A[ni][nj];
            }
        }

        MPI_Send(&sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    
    if (world_rank == 0) {
        std::string ans;
       // std::cout << std::unitbuf;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                ans += std::to_string(A[i][j]);
                ans += " ";
            }
        }

        std::cout << ans;
    }

    MPI_Finalize();
}
