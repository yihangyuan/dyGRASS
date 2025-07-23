#include <iostream>
#include <tuple>
#include <fstream>
#include <cmath>
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



// TODO: create unorderd map in here
// input need to be 1 based
CSRGraph::CSRGraph(const char* filename, bool is_reverse, long skip_head, int weightFlag) {
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
        vertex_t src = atoi(ss + curr) - 1;
        if (weightFlag) checkwt++;  // Track weight presence if needed
        if (checkwt != 3) {         // Skip weights for max/min tracking
            v_max = max(v_max, src);
            v_min = min(v_min, src);
        }
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        // vertex_t dest = atoi(ss + curr);
        if (checkwt != 3) edge_count++;  // Count edges (ignore weights)
        if (checkwt == 3) checkwt = 0;
    }

    // Initialize the CSRGraph structure with calculated values

    this->line_count = edge_count>>1;
    this->edge_count = is_reverse ? edge_count : edge_count>>1;
    this->vert_count = v_max + 1;
    this->v_max = v_max;
    this->v_min = v_min;
    int digit_num = findDigitsNum(vert_count);
    this->multiplier = pow(10, digit_num);

    // Resize adjacency list, weight list, begin list, and degree array based on max vertex ID
    this->begin.resize(this->vert_count + 1, 0);
    this->adj.resize(this->edge_count);
    this->weight.resize(this->edge_count);
    this->degree.resize(this->vert_count, 0);  // No resizing needed during degree population
    this->from.resize(this->edge_count);
    this->mtx.resize(line_count);
    this->reverse.resize(this->edge_count);
    this->edge_map.reserve((unsigned)(this->line_count)*1.5);

    // Pass 2: Populate degrees
    curr = next = 0;
    while (next < file_size) {
        vertex_t src = atoi(ss + curr) - 1;
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        curr = next;
        vertex_t dest = atoi(ss + curr) - 1;
        
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;

        if(weightFlag != 0) { // Skip weight if present
            while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
            while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        }

        curr = next;
        this->degree[src]++;
        if (is_reverse) this->degree[dest]++; 
    }
    
    // Compute cumulative sums to populate begin array
    this->begin[0] = 0;
    for (size_t i = 1; i <= this->vert_count; i++) {
        this->begin[i] = this->begin[i - 1] + this->degree[i - 1];
    }

    // Reset degree counters for use in adjacency list population
    std::fill(this->degree.begin(), this->degree.end(), 0);

    // Pass 3: Populate adjacency list using begin array
    curr = next = 0;
    size_t offset = 0;
    while (offset < line_count) { 
        vertex_t src = atoi(ss + curr) - 1;
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;

        vertex_t dest = atoi(ss + curr) - 1;
        // Populate adjacency list
        this->adj[this->begin[src] + this->degree[src]] = dest;
        this->from[this->begin[src] + this->degree[src]] = offset;
        this->reverse[this->begin[src] + this->degree[src]] = this->degree[dest];
        if (is_reverse) {
            if (dest == src){
                this->adj[this->begin[dest] + this->degree[dest] + 1] = src;
                this->from[this->begin[dest] + this->degree[dest] + 1] = offset;
                this->reverse[this->begin[dest] + this->degree[dest] + 1] = this->degree[src];
            }
            else{
                this->adj[this->begin[dest] + this->degree[dest]] = src;
                this->from[this->begin[dest] + this->degree[dest]] = offset;
                this->reverse[this->begin[dest] + this->degree[dest]] = this->degree[src];
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

        this->weight[this->begin[src] + this->degree[src]] = wtvalue;
        if (is_reverse) this->weight[this->begin[dest] + this->degree[dest]] = wtvalue;

        this->mtx[offset] = make_tuple(src, dest, wtvalue);

        long a = src, b = dest;
        int degree_a = this->degree[a], degree_b = this->degree[b];
        if (a > b){
            swap(a, b);
            swap(degree_a, degree_b);   
        } 
        long key = a * this->multiplier + b;
        pair<index_t,index_t>value = {degree_a, degree_b};
        this->edge_map.insert({key, value});
        // cout << "try to access edge_map: " << this->edge_map.at(key).first << " " << this->edge_map.at(key).second << endl;
        // cout << "try to access edge_map: " << this->edge_map.count(key) << endl;

        // set the next position for next iteration
        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        this->degree[src]++;
        if (is_reverse) this->degree[dest]++;
        
        offset++;
    }
    // long key = 65042065240;
    // cout << "before test" << endl;
    // // cout << "try to access edge_map: " << this->edge_map.at(key).first << " " << this->edge_map.at(key).second << endl;
    // cout << "try to access edge_map: " << this->edge_map.count(key) << endl;
    munmap(ss_head, file_size);
    close(fd);

    this->adj_list = this->adj.data();
    this->weight_list = this->weight.data();
    this->beg_pos = this->begin.data();
    this->degree_list = this->degree.data();

}

void CSRGraph::remove_edge_from_targets(){
    
}



//  require input mtx is 1 based
// split to 10 parts for 10 times experiments
void CSRGraph::read_targets(const char* filename, int process_line, int weightFlag, int divisor) {

    this->divisor = divisor;

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
    weight_t* weights = nullptr;
    if (weightFlag != 0) {
        weights = new weight_t[edge_count];
    }
    // Second pass to populate source and destination arrays 
    vertex_t source, dest;
    size_t offset = 0;
    curr = 0;
    next = 0;

   
    int interval = (edge_count + divisor - 1) / divisor;
    int initial_interval = interval;
    weight_t wtvalue;
    int write_location = 0;
    while (offset < edge_count) {
        char* sss = ss + curr;
        source = atoi(sss) - 1; // Convert to 0-based index
        // source = atoi(sss); // already converted in Julia side, decremental ONLY!

        while (ss[next] != ' ' && ss[next] != '\n' && ss[next] != '\t') next++;
        while (ss[next] == ' ' || ss[next] == '\n' || ss[next] == '\t') next++;
        curr = next;
        char* sss1 = ss + curr;
        dest = atoi(sss1) - 1; // Convert to 0-based index
        // dest = atoi(sss1); // already converted in Julia side, decremental ONLY!

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
        if ((write_location % interval == 0) && (interval == initial_interval)) {

            write_location += interval;

            if (write_location + interval* 2 > edge_count * 2) {

                interval = edge_count - write_location/ 2;
            }
        }
        
    }

    // Clean up mapped memory and file descriptor
    munmap(ss_head, file_size);
    close(fd);


    this->sample_weights = weights;
    this->sample_nodes = ext;
    this->sample_count = edge_count;

}




