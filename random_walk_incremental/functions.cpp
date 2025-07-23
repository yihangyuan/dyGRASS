#include<iostream>
#include<tuple>
#include<fstream>
#include <sys/stat.h> // for stat
#include <sys/mman.h> // for mmap
#include <fcntl.h> // for open
#include <assert.h> // for assert
#include <vector>
#include <unistd.h> // for close
#include "functions.h"
using namespace std;



inline off_t fsize(const char *filename) {
    struct stat st; // a stat structure for store file information
    if (stat(filename, &st) == 0) // read file information to st, 0 means success
        return st.st_size;
    return -1; 
}

// NOTE: Right now, input graph is 0-based !
CSRGraph read_graph(const char* filename, bool is_reverse, long skip_head, int weightFlag) {
    int fd = open(filename, O_RDONLY);  // Open file in read-only mode
    if (fd == -1) {
        perror("Error opening file");
        exit(-1);
    }

    size_t file_size = fsize(filename);  // Get the file size
    char* ss_head = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(ss_head != MAP_FAILED);  // Check if mmap succeeded
    madvise(ss_head, file_size, MADV_SEQUENTIAL);  // Hint for sequential access

    // Skip the first few lines based on skip_head parameter
    size_t head_offset = 0;
    int skip_count = 0;
    while (skip_count < skip_head && head_offset < file_size) {
        if (ss_head[head_offset++] == '\n') skip_count++;
    }

    char* ss = ss_head + head_offset;
    file_size -= head_offset;
    size_t curr = 0, next = 0, edge_count = 0;
    vertex_t v_max = 0, v_min = INFTY;
    int checkwt = 0;
    
  
    // Pass 1: Calculate edge count, max/min vertices without populating degrees
    while (next < file_size) {
        vertex_t src = atoi(ss + curr);
        if (weightFlag) checkwt++;  // Track weight presence if needed
        if (checkwt != 3) {         // Skip weights for max/min tracking
            v_max = max(v_max, src);
            v_min = min(v_min, src);
        }
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        vertex_t dest = atoi(ss + curr);
        if (checkwt != 3) edge_count++;  // Count edges (ignore weights)
        if (checkwt == 3) checkwt = 0;
    }

    // Initialize the CSRGraph structure with calculated values
    CSRGraph graph;
    const index_t line_count = edge_count>>1;
    graph.edge_count = is_reverse ? edge_count : edge_count>>1;
    graph.vert_count = v_max + 1;
    graph.v_max = v_max;
    graph.v_min = v_min;

    // Resize adjacency list, weight list, begin list, and degree array based on max vertex ID
    graph.begin.resize(graph.vert_count + 1, 0);
    graph.adj.resize(graph.edge_count);
    graph.weight.resize(graph.edge_count);
    graph.degree.resize(graph.vert_count, 0);  // No resizing needed during degree population
    graph.from.resize(graph.edge_count);
    graph.mtx.resize(line_count);
    graph.reverse.resize(graph.edge_count);

    // Pass 2: Populate degrees
    curr = next = 0;
    while (next < file_size) {
        vertex_t src = atoi(ss + curr);
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        curr = next;
        vertex_t dest = atoi(ss + curr);
        
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        if(weightFlag != 0) { // Skip weight if present
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }

        curr = next;
        graph.degree[src]++;
        if (is_reverse) graph.degree[dest]++; 
    }
    
    // Compute cumulative sums to populate begin array
    graph.begin[0] = 0;
    for (size_t i = 1; i <= graph.vert_count; i++) {
        graph.begin[i] = graph.begin[i - 1] + graph.degree[i - 1];
    }

    // Reset degree counters for use in adjacency list population
    std::fill(graph.degree.begin(), graph.degree.end(), 0);

    // Pass 3: Populate adjacency list using begin array
    curr = next = 0;
    size_t offset = 0;
    while (offset < line_count) { 
        vertex_t src = atoi(ss + curr);
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        vertex_t dest = atoi(ss + curr);
        // Populate adjacency list
        graph.adj[graph.begin[src] + graph.degree[src]] = dest;
        graph.from[graph.begin[src] + graph.degree[src]] = offset;
        graph.reverse[graph.begin[src] + graph.degree[src]] = graph.degree[dest];
        if (is_reverse) {
            if (dest == src){
                graph.adj[graph.begin[dest] + graph.degree[dest] + 1] = src;
                graph.from[graph.begin[dest] + graph.degree[dest] + 1] = offset;
                graph.reverse[graph.begin[dest] + graph.degree[dest] + 1] = graph.degree[src];
            }
            else{
                graph.adj[graph.begin[dest] + graph.degree[dest]] = src;
                graph.from[graph.begin[dest] + graph.degree[dest]] = offset;
                graph.reverse[graph.begin[dest] + graph.degree[dest]] = graph.degree[src];
            }
        }


        weight_t wtvalue;
        if (weightFlag != 0) {
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
            curr = next;
            wtvalue = atof(ss + curr);
        }
        else {
            wtvalue = 1.0;
        }
        graph.weight[graph.begin[src] + graph.degree[src]] = wtvalue;
        if (is_reverse) graph.weight[graph.begin[dest] + graph.degree[dest]] = wtvalue;

        graph.mtx[offset] = make_tuple(src, dest, wtvalue);

        // set the next position for next iteration
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        graph.degree[src]++;
        if (is_reverse) graph.degree[dest]++;
        
        offset++;
    }

    munmap(ss_head, file_size);
    close(fd);

    return graph;  // Return CSRGraph containing graph data in memory
}


// int* read_target(const char* filename, size_t& target_count) {

// }

//  require input list is 1 based
// split to 10 parts for 10 times experiments
Targets read_targets(const char* filename, int process_line, int weightFlag) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("File open error");
        exit(-1);
    }

    size_t file_size = fsize(filename);
    char* ss_head = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    assert(ss_head != MAP_FAILED);
    madvise(ss_head, file_size, MADV_SEQUENTIAL);

    char* ss = ss_head;
    size_t curr = 0, next = 0;
    size_t edge_count = 0;
    int checkwt = 0;

    // First pass to count edges
    while (next < file_size) {
        if (weightFlag != 0) checkwt++;

        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        if (checkwt != 3) edge_count++;
        if (checkwt == 3) checkwt = 0;
    }

    const index_t line_count = edge_count >> 1;
    edge_count = edge_count >> 1;
    if (process_line != -1) {
        edge_count = process_line;
    }

    // Allocate memory for edge data in a single array
    index_t* ext = new index_t[edge_count * 2];
    weight_t* weights = new weight_t[edge_count];
    // Second pass to populate source and destination arrays 
    vertex_t source, dest;
    size_t offset = 0;
    curr = 0;
    next = 0;

    int divisor = 10;
    int interval = (edge_count + divisor - 1) / divisor;
    int initial_interval = interval;
    weight_t wtvalue;
    int write_location = 0;
    while (offset < edge_count) {
        char* sss = ss + curr;
        source = atoi(sss) - 1; // Convert to 0-based index

        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        char* sss1 = ss + curr;
        dest = atoi(sss1) - 1; // Convert to 0-based index

        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        

        if (weightFlag != 0) {
            curr = next;
            char* sss1 = ss + curr;
            wtvalue = atof(sss1);
            weights[offset] = wtvalue;

            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }
        curr = next;

        // Store source and destination in the array
        ext[write_location] = source;
        ext[write_location + interval] = dest;

        offset++;
        write_location ++;
        if ((write_location % interval == 0) && (interval == initial_interval))  {
            write_location += interval;
            if (write_location + interval* 2 > edge_count * 2) {

                interval = edge_count - write_location/ 2;
            }
        }
    }

    // Clean up mapped memory and file descriptor
    munmap(ss_head, file_size);
    close(fd);

    // Return the struct containing the edges and edge count
    return {ext, weights, edge_count};
}



