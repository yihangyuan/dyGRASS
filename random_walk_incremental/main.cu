#include <stdio.h>
#include <string.h>
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



#define max_steps 100
using namespace std;

__global__ void
random_walk(
			Random_walk_data * all_data,
			gpu_graph * G, // pass by value, because data inside already on GPU, only pointer to data is passed
			unsigned int n_samples, 
			float distortion
			){

	unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;

	curandState local_state = all_data->global_state[tid];
	curand_init(tid, 10, 7, &local_state); 


	unsigned int sourceIndex = G->sample_nodes_device[blockIdx.x]; // first half
	unsigned int targetIndex = G->sample_nodes_device[blockIdx.x+n_samples]; // second half
	float edge_weight = G->sample_weights_device[blockIdx.x];
	unsigned int currentVertex = sourceIndex;

	unsigned int previousVertex = 0;
	int targetFoundAt = -1;
	int step_count = 1; // +1 for the source
	// unsigned int path [max_steps*4];

	float total_R = 0;
	
	// Select the first neighbor (randomly)
	// Shouldn't have the case of degree = 0
	unsigned int degree = G->degree_list_device[currentVertex];
	float * offset = G->beg_ptr_device[currentVertex];
	unsigned int next = curand(&local_state) % degree;

	unsigned int nextVertex = static_cast<unsigned int>(offset[next]);
	float weight = offset[next + degree];
	total_R += 1/weight;


	if (nextVertex != targetIndex){
		
		previousVertex = currentVertex;
		currentVertex = nextVertex;
		step_count++;

		while(step_count < max_steps ){

			degree = G->degree_list_device[currentVertex];
			if (degree == 1){
				break; // reach a leaf, no other choice
			}

			offset = G->beg_ptr_device[currentVertex];
			unsigned int next = curand(&local_state) % (degree - 1);
			weight_t inter = G->beg_ptr_device[currentVertex][next];
			nextVertex = static_cast<unsigned int>(inter);

			// if next vertex is the previous vertex, select the last vertex
			if (nextVertex == previousVertex){

				nextVertex = static_cast<unsigned int>(offset[degree - 1]);
				weight = offset[2*degree - 1];
			}
			else{
				weight = offset[next + degree];
			}

			total_R += 1/weight;

			if (total_R * edge_weight >= distortion){
				break; // reach the distortion
			}

			if (nextVertex == targetIndex){
				targetFoundAt = step_count;
				break; // reach the target
			}
		
			previousVertex = currentVertex;
			currentVertex = nextVertex;
			step_count++;
		}

	}
	else{ 
		targetFoundAt = step_count; 
	} // target is the first neighbor

	// calculate the conductance if the target is found
	if (targetFoundAt != -1){
		// if the target is not found, set the targetFoundAt to the last step
		all_data->conductance_shared_mem[tid] = 1/total_R;
	}else{
		all_data->conductance_shared_mem[tid] = -1;
	}
	__syncthreads();

	// perform reduction to find the maximum conductance
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

			all_data->final_conductance_device[blockIdx.x] = 1/total_R;

		}
	}
	
};





int main(int argc, char *argv[])
{
    if(argc != 4){cout<<"Input: ./incremental_update <1.graph name> <2.distortion_threshold> <3.# of walkers>\n";exit(0);}
    
	const char* graph_name = argv[1];
	string graph_file = "../dataset/" + string(graph_name) + "/adj_sparse.mtx";
	string target_file = "../dataset/" + string(graph_name) + "/ext.mtx";
	string output_file = "../dataset/" + string(graph_name) + "/updated_adj_sparse.mtx";
    float distortion = atof(argv[2]);
    int n_walkers = atoi(argv[3]); // 512 maximum number of walkers per sample, should be multiple of 32

	

    CSRGraph graph = read_graph(graph_file.c_str(), true, 0, 1);
    Targets targets = read_targets(target_file.c_str(), -1, 1);
	cout << "after read_targets" << endl;
    graph.sample_count = targets.target_count;
    graph.sample_nodes = targets.nodes;
	graph.sample_weights = targets.weights;
    graph.to_pointer();

	// setup the graph, create the pointer for both host and device
	gpu_graph ggraph(graph, distortion);
	gpu_graph * host_graph = &ggraph;
	gpu_graph * device_graph;
	HRR(cudaMalloc(&device_graph, sizeof(gpu_graph)));
	
	// setup the kernel parameters
	unsigned int n_samples = ggraph.sample_count; 
	unsigned int n_threadsPerBlock = n_walkers; // this should be multiple of 32
	unsigned int n_blocksPerGrid = n_samples; 
	int n_threads = n_blocksPerGrid * n_threadsPerBlock;
	// printf("threadsPerBlock: %d, blocksPerGrid: %d, n_threads: %d\n", n_threadsPerBlock, n_blocksPerGrid, n_threads);

	// setup random walk data and copy to device
	Random_walk_data * host_data = new Random_walk_data(
		n_samples, 
		n_threads);
	Random_walk_data * device_data;

	HRR(cudaMalloc(&device_data, sizeof(Random_walk_data)));
	HRR(cudaMemcpy(device_data, host_data, sizeof(Random_walk_data), cudaMemcpyHostToDevice));
	ggraph.copyToDevice();
	HRR(cudaMemcpy(device_graph, host_graph, sizeof(gpu_graph), cudaMemcpyHostToDevice));
	cout<< "Graph copied to GPU" << endl;

	// Create CUDA events

	auto start = std::chrono::high_resolution_clock::now();
	random_walk<<<n_blocksPerGrid, n_threadsPerBlock>>>(device_data, device_graph, n_samples, distortion);
	checkCudaErrors(cudaDeviceSynchronize());
	host_data->get_result();
	host_graph->updateGraphFromResult(
		// host_data->path_selected, 
		host_data->path_selected_flag, 
		host_data->final_conductance 
		// n_steps
	);


	bool sample_nodes_exist = true;
	int count = 0;
	while (sample_nodes_exist){

		sample_nodes_exist = host_graph->updateSampleEdges();
		n_samples = host_graph->sample_count;
		random_walk<<<n_samples, n_threadsPerBlock>>>(device_data, device_graph, n_samples, distortion);
		host_data->get_result();
		host_graph->updateGraphFromResult(
			// host_data->path_selected, 
			host_data->path_selected_flag, 
			host_data->final_conductance 
			// n_steps
		);
		
		float processed = (float)(host_graph->sample_count + host_graph->sample_ptr/2)/host_graph->sample_total_count * 100;
		float extra_memory_usage = (float)(host_graph->extra_offset/4)/host_graph->edge_count * 100;

		count++;
		if (count > 10){
			// debug purpose
			cout << "Reached the maximum number of iterations" << endl;
			break;
		}
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;


	cout << "No path %: " << (float)host_graph->no_path_count/host_graph->sample_total_count * 100 << endl;
	host_graph->saveCurrentMtx(output_file.c_str());
	

	delete host_data;
	HRR(cudaFree(device_data));
	HRR(cudaFree(device_graph));
   return 0;
}