
#ifndef FUNCTION_H
#define FUNCTION_H

#include<iostream>
#include<tuple>
#include<fstream>
#include <sys/stat.h> // for stat
#include <sys/mman.h> // for mmap
#include <fcntl.h> // for open
#include <assert.h> // for assert
#include <vector>
#include <unistd.h> // for close
#include <unordered_map>

using namespace std;

#define INFTY int(1<<30)

typedef unsigned int vertex_t;
typedef unsigned int index_t;
typedef float weight_t;

struct Targets {
    vertex_t* nodes;     // Pointer to the array of edges
    weight_t* weights;   // Pointer to the array of weights
    // first half is source nodes, second half is target nodes

    size_t target_count;  // Number of edges

    ~Targets() {

        delete[] nodes;
        delete[] weights;
        cout <<"Targets deleted" << endl;
    }
};

class CSRGraph {
    public:
        // Graph data
        vector<vertex_t> adj;//c        // Adjacency list storing neighbors vertices
        vector<weight_t> weight;//c     // Weights for each edge
        vector<index_t> begin;//c       // Beginning index for each vertex in the adjacency list
        vector<index_t> degree;//c      // Degree of each vertex
        vector<vertex_t> from;//c 
        vector<vertex_t> reverse;//c
        vector<tuple<vertex_t, vertex_t, weight_t>> mtx; //c
        unordered_map<long, pair<index_t,index_t>> edge_map;
        size_t line_count;//c           // Total number of mtx lines
        size_t edge_count;//c           // Total number of edges
        size_t vert_count;//c           // Total number of vertices
        vertex_t v_max;//c              // Maximum vertex ID in the graph
        vertex_t v_min;//c              // Minimum vertex ID in the graph
        int multiplier;//c
        // Graph data pointer
        vertex_t* adj_list;
        weight_t* weight_list;
        index_t* beg_pos;
        vertex_t* degree_list;
        
        // Target data
        size_t sample_count;//c         // Total number of target vertices
        vertex_t* sample_nodes;//c      // Sample nodes for random walk
        weight_t* sample_weights;//c    // Sample weights for random walk
        int divisor;

        


    CSRGraph() : edge_count(0), vert_count(0), sample_count(0), v_max(0), v_min(INFTY), sample_nodes(nullptr) {}

    CSRGraph(const char* filename, bool is_reverse, long skip_head, int weightFlag);

    ~CSRGraph() {
        cout <<"CSRGraph deleted" << endl;
    }

    void to_pointer(){
        adj_list = adj.data();
        weight_list = weight.data();
        beg_pos = begin.data();
        degree_list = degree.data();
    }

    int findDigitsNum(int n){
        int count = 0;
        while (n != 0){
            n = n/10;
            count++;
        }
        return count;
    }
    
    void read_targets(const char* filename, int process_line, int weightFlag, int divisor);

    // revis graph by removing the edge from the target
    void remove_edge_from_targets();

    void test(){
        int vertex;
        cout << "Test begin, input a number between 0 and " << vert_count<< endl;
        cin >> vertex;
        while (vertex >=0 && vertex < vert_count){
            cout << "Vertex: " << vertex << endl;
            cout << "Degree: " << degree[vertex] << endl;
            cout << "Begin: " << begin[vertex] << endl;
            cout << "Adj: ";
            for (int i = begin[vertex]; i < begin[vertex+1]; i++){
                cout << adj[i] << " ";
            }
            cout << endl;

            cout << "Reverse: " << endl;
            for (int i = begin[vertex]; i < begin[vertex+1]; i++){
                cout << adj[begin[adj[i]] + reverse[i]] << " ";
            }
            cout << endl;
            cin >> vertex;
        }

    }

};





// CSRGraph read_graph(const char* filename, bool is_reverse, long skip_head, int weightFlag);

// tuple<float**, float*> createNeighborArray(const CSRGraph& graph);


#endif