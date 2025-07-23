//10/03/2016
//Graph data structure on GPUs
#ifndef _GPU_GRAPH_H_
#define _GPU_GRAPH_H_
#include <iostream>
// #include <curand.h>
#include "header.h"
// #include "util.h"
#include "herror.h"
// #include "graph.h"
#include "functions.h"
#include <curand_kernel.h>
#include <assert.h>

using namespace std;

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", \
        cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

#define H_ERR( err )(HandleError( err, __FILE__, __LINE__ ))


class gpu_graph
{
	public:
        //Size information
		index_t vert_count;
		index_t edge_count;
        index_t sample_total_count;
		index_t sample_count; 
        index_t no_path_count;
        index_t sample_ptr;
        index_t ext_size;

        // Host pointers
        weight_t * mtx;
        weight_t * ext;
        weight_t ** beg_ptr;
        weight_t * neighbors_data;
        vertex_t * degree_list;
        vertex_t * sample_nodes_all;
        weight_t * sample_weights;
        weight_t * extra_neighbors_data;
        

        // Device pointers
        weight_t ** beg_ptr_device;
        weight_t * neighbors_data_device;
        vertex_t * degree_list_device;
        vertex_t * sample_nodes_device;
        weight_t * sample_weights_device;
        weight_t * extra_neighbors_data_device;

        // Device pointers but generated from host;
        weight_t ** beg_ptr_device_content;
		unsigned  extra_offset;

        //other
        float distortion;

        

	public:
        gpu_graph(
			CSRGraph ginst,
            float distortion
        )
		{   
            this->distortion = distortion;

            this->vert_count = ginst.vert_count;
			this->edge_count = ginst.edge_count;
            this->sample_total_count = ginst.sample_count;
            this->sample_count = (ginst.sample_count + 10 -1)/10;
            // this->ext_size = 0;
            this->no_path_count = 0;
            this->extra_offset = 0;

            auto [mtx, ext] = createMtx(ginst.mtx);
            this->mtx = mtx;
            this->ext = ext;
            this->ext_size = 0;

            //beg_ptr and neighbors_data are allocated on the host
            auto [beg_ptr, neighbors_data] = createNeighborArray(ginst);
            this->beg_ptr = beg_ptr;
            this->neighbors_data = neighbors_data;

            this->degree_list = ginst.degree_list;
            this->sample_nodes_all = ginst.sample_nodes;
            this->sample_ptr = 0;
            this->sample_weights = ginst.sample_weights;

            this->beg_ptr_device_content = new weight_t*[vert_count];
            this->extra_neighbors_data = neighbors_data + edge_count*4;
            
            
        }

        tuple<weight_t*, weight_t*> createMtx(vector<tuple<vertex_t, vertex_t, weight_t>> mtx){
            int len = mtx.size();
            weight_t * mtx_data = new weight_t[len*3];

            for (int i = 0; i < len; i += 1){
                mtx_data[i*3] = get<0>(mtx[i]);
                mtx_data[i*3 + 1] = get<1>(mtx[i]);
                mtx_data[i*3 + 2] = get<2>(mtx[i]);
            }

            weight_t * ext_data = new weight_t[sample_total_count*3];
            return {mtx_data, ext_data};
        }

        tuple<weight_t**, weight_t*> createNeighborArray(const CSRGraph& graph) {
            // Allocate an array of pointers, one for each vertex
            weight_t** beg_ptr = new weight_t*[graph.vert_count];

            weight_t* neighbors_data = new weight_t[graph.edge_count * 4 * 2]; // Allocate space for both indices, weights and from
            size_t offset = 0;
            for (size_t i = 0; i < graph.vert_count; i++) {
                // Set the pointer to the beginning of the adjacency list for the vertex
                beg_ptr[i] = neighbors_data + offset;
                // Copy the adjacency list for the vertex
                index_t degree = graph.degree[i];
                for (size_t j = graph.begin[i]; j < graph.begin[i + 1]; j++) {
                    neighbors_data[offset] = static_cast<weight_t>(graph.adj[j]);
                    neighbors_data[offset + degree] = graph.weight[j];
                    neighbors_data[offset + 2 * degree] = static_cast<weight_t>(graph.from[j]);
                    neighbors_data[offset + 3 * degree] = static_cast<weight_t>(graph.reverse[j]);
                    offset++;
                }
                offset += 3 * degree;

            }

            return {beg_ptr, neighbors_data};

        }

        
        void copyToDevice(){
            // allocation
            HRR(cudaMalloc((void ***)&beg_ptr_device, sizeof(weight_t*)*vert_count)); // need to determine the content
            HRR(cudaMalloc((void **)&neighbors_data_device, sizeof(weight_t)*edge_count*4 * 2));
            HRR(cudaMalloc((void **)&degree_list_device, sizeof(vertex_t)*vert_count));
            HRR(cudaMalloc((void **)&sample_nodes_device, sizeof(vertex_t)*sample_count*2));
            HRR(cudaMalloc((void **)&sample_weights_device, sizeof(weight_t)*sample_count));
            // HRR(cudaMalloc((void **)&extra_neighbors_data_device, sizeof(weight_t)*edge_count*4));
            
            // calculate the content of beg_ptr_device after neighbors_data_device is allocated
            // cout << "before beg_ptr_device_content" << endl;
            unsigned offset = 0;
            for (int i = 0; i < vert_count; i++){
                weight_t * ptr = neighbors_data_device + offset;
                // cout << "ptr: " << ptr << endl;
                beg_ptr_device_content[i] = reinterpret_cast<weight_t*>(ptr);
                // cout << "beg_ptr_device_content[i]: " << beg_ptr_device_content[i] << endl;
                offset += degree_list[i]*4;
                // cout << "offset: " << offset << endl;
            }
            // cout << "after beg_ptr_device_content" << endl;
            // copy
            extra_neighbors_data_device = neighbors_data_device + offset;
            assert(extra_neighbors_data_device == neighbors_data_device + edge_count*4);
            HRR(cudaMemcpy(neighbors_data_device, neighbors_data, sizeof(weight_t)*edge_count*4 * 2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(degree_list_device, degree_list, sizeof(vertex_t)*vert_count, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(beg_ptr_device, beg_ptr_device_content, sizeof(weight_t*)*vert_count, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sample_nodes_device, (sample_nodes_all+sample_ptr), sizeof(vertex_t)*sample_count*2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sample_weights_device, (sample_weights + sample_ptr/2), sizeof(weight_t)*sample_count, cudaMemcpyHostToDevice));

        }

        void updateGraphFromResult(
            // int* path_selected, 
            int* path_selected_flag, 
            float* final_conductance
            // int n_steps
        ){


            // int line_count = edge_count/2;
            // int interval = n_steps*4;
            
            for (int i = 0; i < sample_count; i++){

                int find_at = path_selected_flag[i]; // include source and target
                weight_t weight = sample_weights[sample_ptr/2 + i]; //TODO
                
                // if find the path
                if(find_at != -1){ 

  
                }
                else{ // no path
                    
                    no_path_count += 1;

                    int source = sample_nodes_all[sample_ptr + i];
                    int target = sample_nodes_all[sample_ptr + sample_count + i];
                    // add it to ext
                    // (*ext)[ext_size] = make_tuple(source, target, weight);
                    ext[ext_size*3] = source;
                    ext[ext_size*3 + 1] = target;
                    ext[ext_size*3 + 2] = weight;

                    // manage the extra_neighbors_data
                    int degree_source =  ++ degree_list[source];
                    int degree_target =  ++ degree_list[target];

                    // if (source == target){
                    //     cout << "A"
                    // }

                    manageExtraNeighborsData(source, target, weight, degree_source, degree_target);
                    manageExtraNeighborsData(target, source, weight, degree_target, degree_source);

                    ext_size += 1;
                }
            }
            cout << "no path count: " << no_path_count << endl;

            // copy the updated data to device
            // HRR(cudaMemcpy(neighbors_data_device, extra_neighbors_data_device, sizeof(weight_t)*edge_count*4, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(neighbors_data_device, neighbors_data, sizeof(weight_t)*edge_count*4 * 2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(beg_ptr_device, beg_ptr_device_content, sizeof(weight_t*)*vert_count, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(degree_list_device, degree_list, sizeof(vertex_t)*vert_count, cudaMemcpyHostToDevice));

            // update the sample_ptr
            // track the sample_ptr exceeding the length of sample_nodes_all or not
            // copy the sample_nodes to device

        }

        void manageExtraNeighborsData(int node, int neighbor, weight_t weight, int new_degree, int reverse_index){
            weight_t * ptr_old = beg_ptr[node];
            weight_t * ptr_new = extra_neighbors_data + extra_offset;

            for (int i = 0, j = 0; i < (new_degree - 1) * 4; i++, j++){
                if (j % new_degree == new_degree - 1){ // skip the last one for the new degree
                    j ++;
                }
                ptr_new[j] = ptr_old[i];
            }
            ptr_new[new_degree - 1] = static_cast<weight_t>(neighbor);
            ptr_new[2 * new_degree - 1] = weight;
            ptr_new[3 * new_degree - 1] = static_cast<weight_t>(edge_count/2 + ext_size);
            ptr_new[4 * new_degree - 1] = static_cast<weight_t>(reverse_index - 1);
            beg_ptr[node] = ptr_new;

            weight_t * ptr_new_device = reinterpret_cast<weight_t*>(extra_neighbors_data_device + extra_offset);
            beg_ptr_device_content[node] = ptr_new_device;
            extra_offset += new_degree*4;
        }
            

        bool updateSampleEdges(){
            
            bool not_empty = true;
            sample_ptr += sample_count * 2;
            if (sample_ptr + sample_count * 2 >= sample_total_count * 2){

                sample_count = (sample_total_count * 2 - sample_ptr)/2;
                not_empty = false;
            }


            HRR(cudaMemcpy(sample_nodes_device, (sample_nodes_all+sample_ptr), sizeof(vertex_t)*sample_count*2, cudaMemcpyHostToDevice));
            HRR(cudaMemcpy(sample_weights_device, (sample_weights + sample_ptr/2), sizeof(weight_t)*sample_count, cudaMemcpyHostToDevice));

            return not_empty;
        }


        // NOTE: output is 1 based !
        void saveCurrentMtx(string file_name){
            std::cout << "saving mtx to " << file_name << std::endl;
            std::ofstream txt_file(file_name, std::ios::out);
            if (txt_file.is_open()){
                
                for (int i = 0; i < edge_count/2; i++){
                    txt_file << 
                    static_cast<int>(mtx[i*3]) + 1 << " " << 
                    static_cast<int>(mtx[i*3 + 1]) + 1 << " " << 
                    mtx[i*3 + 2] << endl;
                }

                for (int i = 0; i < ext_size; i++){
                    txt_file << 
                    static_cast<int>(ext[i*3]) + 1 << " " << 
                    static_cast<int>(ext[i*3 + 1]) + 1 << " " << 
                    ext[i*3 + 2] << endl;
                }

                
                txt_file.close();
            }
            else{
                std::cerr << "Unable to open file"<< endl;
            }
            cout << "Updated graph saved to " << file_name << endl;
        }

        ~gpu_graph(){
            cout << "gpu_graph" << endl;
            delete[] beg_ptr;
            delete[] neighbors_data;
            // delete[] degree_list; // don't delete this, it is from the graph
            // delete[] extra_neighbors_data;
            delete[] beg_ptr_device_content;
            delete ext;
            delete mtx;

            cudaFree(beg_ptr_device);
            cudaFree(neighbors_data_device);
            cudaFree(degree_list_device);
            cudaFree(sample_nodes_device);
            cudaFree(sample_weights_device);
            // cudaFree(extra_neighbors_data_device);
            cout << "gpu_graph deleted" << endl;
        }

            


};

class Random_walk_data{

    public:
    int sample_num;
    int n_steps;
    // int* path_selected; // selected path 
	// int* path_selected_device; 
	int* path_selected_flag; // identify weather the path a found
	int* path_selected_flag_device; 
    float* conductance_shared_mem; // store the conductance of each thread for each path
	int* conductance_index_shared_mem; // store thread index to perform reduction to find the maximum conductance
    float* final_conductance; // store the final conductance of each path
    float* final_conductance_device;

    curandState *global_state;
    
    Random_walk_data(
        // int n_steps, 
        int n_samples, 
        int n_threads 
        ){
        this->sample_num = n_samples;
        // this->n_steps = n_steps;
        // path_selected = (int *)malloc(sizeof(int)*n_steps*n_samples * 4); // first half for path, second half for from index
	    path_selected_flag = (int *)malloc(sizeof(int)*n_samples);
        final_conductance = (float *)malloc(sizeof(float)*n_samples);


        HRR(cudaMalloc(&global_state, n_threads * sizeof(curandState)));
        // HRR(cudaMalloc((void **) &path_selected_device, sizeof(int)*n_steps*n_samples * 4)); // path, from, next, reverse
        HRR(cudaMalloc((void **) &path_selected_flag_device, sizeof(int)*n_samples));
        HRR(cudaMalloc((void **) &conductance_shared_mem, sizeof(float)*n_threads));
        HRR(cudaMalloc((void **) &conductance_index_shared_mem, sizeof(int)*n_threads));
        HRR(cudaMalloc((void **) &final_conductance_device, sizeof(float)*n_samples));
    };


    void get_result(){
        // HRR(cudaMemcpy(path_selected, path_selected_device, sizeof(int)*n_steps*sample_num * 4, cudaMemcpyDeviceToHost));
        HRR(cudaMemcpy(path_selected_flag, path_selected_flag_device, sizeof(int)*sample_num, cudaMemcpyDeviceToHost));
        HRR(cudaMemcpy(final_conductance, final_conductance_device, sizeof(float)*sample_num, cudaMemcpyDeviceToHost));
    }
    
    ~Random_walk_data(){
        HRR(cudaFree(global_state));
        // free(path_selected);
        free(path_selected_flag);
        free(final_conductance);
        // HRR(cudaFree(path_selected_device));
        HRR(cudaFree(path_selected_flag_device));
        HRR(cudaFree(conductance_shared_mem));
        HRR(cudaFree(conductance_index_shared_mem));
        HRR(cudaFree(final_conductance_device));
        cout <<"Random walk data deleted" << endl;
    };

};
#endif
