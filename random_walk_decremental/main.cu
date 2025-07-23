#include <stdio.h>
#include <string.h>
// #include <mpi.h>
#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <curand.h>
#include <unistd.h>
#include <errno.h>
#include <netdb.h>
#include <chrono>
#include "functions.h"
#include "gpu_graph.cuh"
#include "helper_cuda.h"

// #include "sampler.cuh"

#define max_steps 100
using namespace std;


__global__ void
random_walk_decremental(
			Random_walk_data_decremental * all_data,
			GPU_Dual_Graph * G, // pass by value, because data inside already on GPU, only pointer to data is passed
			unsigned int n_samples, 
			int n_steps
			){

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	curandState local_state = all_data->global_state[tid];
	curand_init(tid, 10, 7, &local_state);

	unsigned int sourceIndex = G->decremental_sample_nodes_device[blockIdx.x]; // first half
	unsigned int targetIndex = G->decremental_sample_nodes_device[blockIdx.x+n_samples]; // second half
	unsigned int currentVertex = sourceIndex;
	unsigned int previousVertex = 0;
	int targetFoundAt = -1;
	int step_count = 1; // +1 for the source
	unsigned int path [max_steps];
	float total_R = 0;

	path[0] = sourceIndex;
	unsigned int degree = G->dense_degree_list_device[currentVertex];
	unsigned int original_degree = G->dense_degree_original_device[currentVertex];
	float * offset = G->dense_beg_ptr_device[currentVertex];
	unsigned int next = curand(&local_state) % degree;
	unsigned int nextVertex = static_cast<unsigned int>(offset[next]);
	float weight = offset[next + original_degree];
	total_R += 1/weight;
	path[step_count] = nextVertex;

	if (nextVertex != targetIndex){
		previousVertex = currentVertex;
		currentVertex = nextVertex;
		step_count++;

		while(step_count < n_steps){

			degree = G->dense_degree_list_device[currentVertex];
			if (degree == 1){
				break; // reach a leaf, no other choice
			}
			if (degree == 0) {
				// throw a error
				printf("Error: degree is 0\n");
			}

			original_degree = G->dense_degree_original_device[currentVertex];
			offset = G->dense_beg_ptr_device[currentVertex];
			unsigned int next = curand(&local_state) % (degree - 1);
			nextVertex = static_cast<unsigned int>(offset[next]);

			if (nextVertex == previousVertex){

				nextVertex = static_cast<unsigned int>(offset[degree - 1]);
				weight = offset[degree - 1 + original_degree];
			}
			else{
				weight = offset[next + original_degree];
			}

			path[step_count] = nextVertex;
			total_R += 1/weight;

			if(nextVertex == targetIndex){
				targetFoundAt = step_count;
				break;
			}

			previousVertex = currentVertex;
			currentVertex = nextVertex;
			step_count++;
		}
	}
	else{ targetFoundAt = step_count; } // target is the first neighbor

	if (targetFoundAt != -1){
		all_data->conductance_shared_mem[tid] = 1/total_R;
	}else{
		all_data->conductance_shared_mem[tid] = -1;
	}

	__syncthreads();

	all_data->conductance_index_shared_mem[tid] = threadIdx.x;
	float left, right;
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (threadIdx.x < s) {
			left = all_data->conductance_shared_mem[tid];
			right = all_data->conductance_shared_mem[tid + s];

			bool isSmall = (left < right);
			all_data->conductance_shared_mem[tid] = isSmall ? right : left;
			all_data->conductance_index_shared_mem[tid] = isSmall ? all_data->conductance_index_shared_mem[tid + s]: all_data->conductance_index_shared_mem[tid];
		}
		__syncthreads();
	}
	
	int max_conductance_thread = all_data->conductance_index_shared_mem[blockDim.x *blockIdx.x]; // get the maximum index from the first thread of the block
	if (threadIdx.x == max_conductance_thread){
		all_data->path_selected_flag_device[blockIdx.x] = targetFoundAt;
		if (targetFoundAt > 0){ // >0 means the target is found
			for (int i = 0; i <= targetFoundAt; i++){
				all_data->path_selected_device[blockIdx.x * n_steps + i] = path[i];
			}
			all_data->final_conductance_device[blockIdx.x] = 1/total_R;
		}
	}

}

__global__ void
random_walk_small(
			Random_walk_data_decremental * all_data,
			GPU_Dual_Graph * G,
			unsigned int n_samples
			){
	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	curandState local_state = all_data->global_state[tid];
	curand_init(tid, 10, 7, &local_state);

	unsigned int sourceIndex = G->heuristic_sample_nodes_device[blockIdx.x]; // first half
	unsigned int currentVertex = sourceIndex;
	unsigned int previousVertex = 0;
	unsigned int path [3];


	path[0] = sourceIndex;
	unsigned int degree = G->dense_degree_list_device[currentVertex];
	float * offset = G->dense_beg_ptr_device[currentVertex];
	unsigned int next = curand(&local_state) % degree;
	unsigned int nextVertex = static_cast<unsigned int>(offset[next]);
	path[1] = nextVertex;

	previousVertex = currentVertex;
	currentVertex = nextVertex;

	degree = G->dense_degree_list_device[currentVertex];
	next = curand(&local_state) % degree;
	nextVertex = static_cast<unsigned int>(offset[next]);
	
	if (nextVertex == previousVertex){
		nextVertex = static_cast<unsigned int>(offset[degree - 1]);
	}
	path[2] = nextVertex;


	for (int i = 0; i < 3; i++){

		all_data->path_selected_device[tid * 3 + i] = path[i];
	}
			
}




int main(int argc, char *argv[])
{	
	//cp adj_dense_mix.mtx adj_sparse.mtx del.mtx ./../../.; ../../random_walk_GPU; cp added_edges.mtx ~/Downloads/dyGRASS_UP

    if(argc != 2){cout<<"Input: ./decremental_update <1. graph name>\n";exit(0);}
	const char* graph_name = argv[1];
	string sparse_graph_file = "../dataset/" + string(graph_name) + "/updated_adj_sparse.mtx";
	string dense_graph_file = "../dataset/" + string(graph_name) + "/updated_dense.mtx";
	string del_file = "../dataset/" + string(graph_name) + "/del.mtx";
	string output_file = "../dataset/" + string(graph_name) + "/added_edges.mtx";

    int n_steps = 100;
    int n_walkers = 512;
	int n_samples_splits = 10;
	int n_heuristic_walkers = 16;

    // CSRGraph graph = read_graph(graph_file, true, 0, 1);
	cout << "Read graph and sparsifier" << endl;
	CSRGraph dense_graph(dense_graph_file.c_str(), true, 0, 1);// file_name, is_reverse, skip_head, weightFlag
	CSRGraph sparse_graph(sparse_graph_file.c_str(), true, 0, 1);

	// load the del edges to dense_graph
	cout << "Read targets" << endl;
    dense_graph.read_targets(del_file.c_str(), -1, 0, n_samples_splits); // all_line, no_weight, divisor
	

	// graph.test();
	GPU_Dual_Graph ggraph(dense_graph, sparse_graph);
	size_t old_edge_num = ggraph.sparse_map->size();
	
	GPU_Dual_Graph * host_graph = &ggraph;
	GPU_Dual_Graph * device_graph;
	HRR(cudaMalloc(&device_graph, sizeof(GPU_Dual_Graph)));
	
	
	bool is_empty = host_graph->removeAndUpdateDelEdges();
	host_graph->copyDualGraphToDevice();
	cout<< "Graph copied to GPU" << endl;

	// setup the kernel parameters
	unsigned int n_samples = ggraph.decremental_sample_num; 
	unsigned int n_threadsPerBlock = n_walkers; // this should be multiple of 32
	unsigned int n_blocksPerGrid = n_samples; //TODO: need to change in the for loop
	int n_threads = n_blocksPerGrid * n_threadsPerBlock;



	// cout << "before Random_walk_data" << endl;
	Random_walk_data_decremental * host_data = new Random_walk_data_decremental(n_steps, n_samples, n_threads);
	Random_walk_data_decremental * device_data;
	HRR(cudaMalloc(&device_data, sizeof(Random_walk_data_decremental)));

	HRR(cudaMemcpy(device_data, host_data, sizeof(Random_walk_data_decremental), cudaMemcpyHostToDevice));
	HRR(cudaMemcpy(device_graph, host_graph, sizeof(GPU_Dual_Graph), cudaMemcpyHostToDevice));
	// ggraph.copyToDevice();
	
	
	n_blocksPerGrid = host_graph->filtered_del_sample_count;
	auto start = std::chrono::high_resolution_clock::now();
	random_walk_decremental<<<n_blocksPerGrid, n_threadsPerBlock>>>(device_data, device_graph, n_samples, n_steps);
	checkCudaErrors(cudaDeviceSynchronize());
	host_data->get_result();
	host_graph->incrementalUpdateSparse(host_data->path_selected, host_data->path_selected_flag, host_data->final_conductance, n_steps);
	
	if(host_graph->heuristic_sample_num > 0){
		// small random walk on the heuristic sample nodes
		random_walk_small<<<host_graph->heuristic_sample_num, n_heuristic_walkers>>>(device_data, device_graph, host_graph->heuristic_sample_num);
		checkCudaErrors(cudaDeviceSynchronize());
		// update the sparse graph
		host_data->get_result();
		host_graph->heuristicRecovery(host_data->path_selected, n_heuristic_walkers);

	}


	int count = 0;
	while (! is_empty){

		is_empty = host_graph->removeAndUpdateDelEdges();
		n_samples = host_graph->filtered_del_sample_count;
		random_walk_decremental<<<n_samples, n_threadsPerBlock>>>(device_data, device_graph, n_samples, n_steps);
		checkCudaErrors(cudaDeviceSynchronize());
		host_data->get_result();
		host_graph->incrementalUpdateSparse(host_data->path_selected, host_data->path_selected_flag, host_data->final_conductance, n_steps);

		if(host_graph->heuristic_sample_num > 0){
			// small random walk on the heuristic sample nodes
			random_walk_small<<<host_graph->heuristic_sample_num, n_heuristic_walkers>>>(device_data, device_graph, host_graph->heuristic_sample_num);
			checkCudaErrors(cudaDeviceSynchronize());
			// update the sparse graph
			host_data->get_result();
			host_graph->heuristicRecovery(host_data->path_selected, n_heuristic_walkers);
		}
		count++;
		if(count > n_samples_splits){
			cout << "break by count, check code" << endl;
			break;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

	size_t new_edge_num = host_graph->sparse_map->size();
	cout << "Increased edges number: " << new_edge_num - old_edge_num << endl;
	cout << "Density: " << (float)(new_edge_num - host_graph->vertex_num)/host_graph->vertex_num * 100<< " %" << endl;
	// host_graph->sparse_map_to_1_based_mtx("decremental_updated_sparse.mtx");
	host_graph->save_added_edges(output_file.c_str());
	

	delete host_data;
	HRR(cudaFree(device_data));
	HRR(cudaFree(device_graph));
   return 0;
}