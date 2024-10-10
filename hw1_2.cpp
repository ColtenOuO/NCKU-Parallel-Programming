#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <mpi.h>
#include <algorithm>
#include <stdio.h>

class Coordinate {
    private:
        long long x, y, id;
    public:
        Coordinate(long long x = 0, long long y = 0, long long id = 0) : x(x), y(y), id(id) {}
        long long getX() const { return this -> x; }
        long long getY() const { return this -> y; }
        long long getID() const { return this -> id; }
        
    friend std::istream& operator>>(std::istream& input, Coordinate &coordinate);
};
std::istream& operator>>(std::istream& input, class Coordinate &coordinate) {
    input >> coordinate.x >> coordinate.y;
    return input;
}
long long cross(Coordinate o,Coordinate a,Coordinate b)
{
    return (a.getX() - o.getX()) * (b.getY() - o.getY()) - (a.getY() - o.getY()) * (b.getX() - o.getX());
}
class FileInput {
    protected:
        std::string file_name;
        std::vector<class Coordinate> v;
    public:
        FileInput(std::string file_name) : file_name(file_name) {}
        void Parsing() {
                std::ifstream input_file(this->file_name);
                int n;
                input_file >> n;
                for (int i = 1; i <= n; i++) {
                    Coordinate input(0, 0, i);
                    input_file >> input;
                    v.emplace_back(input);
                }
                input_file.close();
            return;
        }
};
class Solution : private FileInput {
    private:
        std::vector<class Coordinate> result;
    public:
        Solution(int rank, const std::string &filename) : FileInput(filename) {
            if( rank == 0 ) Parsing();
        }
        std::vector<class Coordinate> getVec() { return v; }
        void Sort_Insert(std::vector<class Coordinate> target) {
            std::for_each(target.begin(), target.end(), [this](const Coordinate& coord) {
                (this -> result).emplace_back(Coordinate(coord.getX(), coord.getY(), coord.getID()));
            });

            return;
        }
        std::vector<class Coordinate> Andrew_monotone_chain() {
            std::vector<class Coordinate> answer;
             sort(result.begin(),result.end(), [](const Coordinate &a, const Coordinate &b) {
                if( a.getX() == b.getX() ) return a.getY() < b.getY();
                return a.getX() < b.getX();
            });
            int cnt = 0;
            for(int i=0;i<result.size();i++) // 凸包下半部要選逆時針轉的, 外積 > 0 表示逆時針轉
            {
                while( cnt >= 2 && cross(answer[cnt-2],answer[cnt-1],result[i]) <= 0 ) answer.pop_back(), cnt--;
                answer.emplace_back(result[i]), cnt++;
            }

            int under = cnt;

            for(int i=(int)result.size()-2;i>=0;i--) // 選上半部的時候不要把下半部選好的 pop 走
            {
                while( cnt >= under + 1 && cross(answer[cnt-2],answer[cnt-1],result[i]) <= 0 ) answer.pop_back(), cnt--;
                answer.emplace_back(result[i]), cnt++;
            }

            return answer;
        }
};
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    
    std::string s;
    if( world_rank == 0 ) { std::cin >> s; }
    class Solution solution(world_rank, s); // Init input

    if( world_rank == 0 ) {
        std::vector<class Coordinate> p = solution.getVec();
        long long start_index, end_index;
        int check = 0;
        if( (int)p.size() < 8 ) {
            sort(p.begin(),p.end(), [](const Coordinate &a, const Coordinate &b) {
                if( a.getX() == b.getX() ) return a.getY() < b.getY();
                return a.getX() < b.getX();
            });
            solution.Sort_Insert(p);
            MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
        else {
            check = 1;
            MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::vector<long long> x((int)p.size()), y((int)p.size()), id((int)p.size());
            for(int i=0;i<p.size();i++) x[i] = p[i].getX(), y[i] = p[i].getY(), id[i] = p[i].getID();
            long long p_sz = (long long)p.size();
            MPI_Bcast(&p_sz, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(x.data(), (int)p.size(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(y.data(), (int)p.size(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            MPI_Bcast(id.data(), (int)p.size(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);
            
            
            long long start_index = 0, end_index = 0, block = (long long)p.size() / 8;
            for(int i=1;i<8;i++) {
                start_index = i * block;  // 計算正確的 start_index
                end_index = std::min((i + 1) * block, (long long)p.size());
                MPI_Send(&start_index, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
                MPI_Send(&end_index, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
            }

            start_index = 0, end_index = block;
            std::vector<class Coordinate> p2;
            for(long long i=start_index;i<end_index;i++) p2.emplace_back(p[i]);
            sort(p2.begin(),p2.end(), [](const Coordinate &a, const Coordinate &b) {
                if( a.getX() == b.getX() ) return a.getY() < b.getY();
                return a.getX() < b.getX();
            });

            solution.Sort_Insert(p2);
        }
    }
    
    int check = 0;
    if( world_rank != 0 ) {
        MPI_Bcast(&check, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    if (world_rank != 0 && check == 1) {
        std::vector<class Coordinate> p;
        std::vector<long long> x, y, id;
        long long p_sz = 0;
        MPI_Bcast(&p_sz, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        x.resize(p_sz), y.resize(p_sz), id.resize(p_sz);
        MPI_Bcast(x.data(), p_sz, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(y.data(), p_sz, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(id.data(), p_sz, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        long long start_index = 0, end_index = 0;
        MPI_Recv(&start_index, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&end_index, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (long long i = 0; i < p_sz; i++)
            p.emplace_back(Coordinate(x[i], y[i], id[i]));
        std::vector<class Coordinate> p2;
        for (long long i = start_index; i < end_index; i++)
            p2.emplace_back(p[i]);
        sort(p2.begin(), p2.end(), [](const Coordinate &a, const Coordinate &b) {
            if (a.getX() == b.getX()) return a.getY() < b.getY();
            return a.getX() < b.getX();
        });
        std::vector<long long> x2, y2, id2;
        // Correctly resize arrays
        x2.resize(p2.size()), y2.resize(p2.size()), id2.resize(p2.size());
        // Use the correct loop range
        for (long long i = 0; i < p2.size(); i++)
            x2[i] = p2[i].getX(), y2[i] = p2[i].getY(), id2[i] = p2[i].getID();
        long long sz = x2.size();
        MPI_Send(&sz, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(x2.data(), sz, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(y2.data(), sz, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
        MPI_Send(id2.data(), sz, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    if( world_rank == 0 && solution.getVec().size() >= 8 ) {
        for(int i=1;i<8;i++) {
            std::vector<class Coordinate> v;
            std::vector<long long> x2,y2,id2;
            long long sz;
            MPI_Recv(&sz, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            x2.resize(sz), y2.resize(sz), id2.resize(sz);
            MPI_Recv(x2.data(), sz, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(y2.data(), sz, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(id2.data(), sz, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(long long j = 0; j < sz; j++) v.emplace_back(Coordinate(x2[j], y2[j], id2[j]));
            solution.Sort_Insert(v);
        }
    }

    if( world_rank == 0 ) {
        std::vector<class Coordinate> ans = solution.Andrew_monotone_chain();
        for(long long i=(long long)ans.size()-1;i>=1;i--) std::cout << ans[i].getID() << " ";
    }

    MPI_Finalize();
}
