#ifndef _GPUROLLING_CUH
#define _GPUROLLING_CUH


#include "GPUVector.cuh"

#include <stdio.h>
#include <ctime>
#include <vector>
// CUDA runtime
#include <cuda_runtime.h>

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions

#include <cuda_occupancy.h>

#define NUM_BLOCK     8
#define NUM_THREAD    128
#define STRIDE        1024

namespace GPURolling
{
	typedef Vector3d        Point;
	typedef Vector3d        Normal;
}

struct Parameter
{
	double sigma_s;
	double sigma_r;
	unsigned int iter_num;
} parameter;

class GPUMesh
{
	GPUMesh();
	GPUMesh(GPURolling::Normal *dnormal, GPURolling::Point *dpoint, double *darea, unsigned int num);


	void allocMesh();
	void freeMesh();

public:
	GPURolling::Normal *d_normal;
	GPURolling::Point  *d_centroid;
	double *d_area;
	unsigned int num;
};

__device__ inline double dGaussian(double sigma, double r)
{
	return exp(-0.5 * (r * r) / (sigma * sigma));
}

__global__ void
updateNormalKernel(const GPURolling::Normal *dnormal, const GPURolling::Point *dpoint,
	const double *darea, 		/*parameter*/
	const GPURolling::Normal *dfiltered, GPURolling::Normal *temp,
	const unsigned int num, const double sigma_s, const double sigma_r);
__global__ void 
updateVertexKernel(const GPURolling::Point *d_all_centroid, const GPURolling::Point *d_all_vertex, 
	const GPURolling::Normal *d_filtered_normal,
	const double *d_all_area, GPURolling::Point *new_points,
	const unsigned int face_num, const unsigned int vertex_num,
	double sigma_s, double sigma_r);

extern "C" void
getData(GPURolling::Point *all_centriod, GPURolling::Normal *all_face_normal, double *all_face_area, 
	unsigned int num, GPURolling::Normal **filtered_ptr,double sigma_s, double sigma_r, unsigned int iter)
{
	size_t size_centriod = num * sizeof(GPURolling::Point);
	size_t size_normal = num * sizeof(GPURolling::Normal);
	size_t size_area = num * sizeof(double);

	std::cout << "the mesh size " << num << std::endl;
	std::cout << "start GPU rolling normal filtering " <<std::endl;
	//Allocate the device memory
	GPURolling::Point *d_all_centriod = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_all_centriod, size_centriod));
	GPURolling::Normal *d_all_face_normal = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_all_face_normal, size_normal));
	double *d_all_face_area = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_all_face_area, size_area));
	
	//copy memory from host to device
	checkCudaErrors(cudaMemcpy(d_all_centriod, all_centriod, size_centriod, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_all_face_normal, all_face_normal, size_normal, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_all_face_area, all_face_area, size_area, cudaMemcpyHostToDevice));

	GPURolling::Normal *d_filtered_normal = NULL;
	size_t size = sizeof(GPURolling::Normal) * num;
	checkCudaErrors(cudaMalloc((void**)&d_filtered_normal, size));
	checkCudaErrors(cudaMemset(d_filtered_normal, 0, size));

	GPURolling::Normal *temp_filtered = NULL;
	checkCudaErrors(cudaMalloc((void**)&temp_filtered, size));

	// Occupancy 
	std::cout << "calculate occupancy" << std::endl;

	int numBlocks;
	int blockSize = 32;
	
	// These variables are used to convert occupancy to warps
	int device;
	cudaDeviceProp prop;
	int activeWarps;
	int maxWarps;

	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&numBlocks,
		updateNormalKernel,
		blockSize,
		0);

	activeWarps = numBlocks * blockSize / prop.warpSize;
	maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
	std::cout << "num of blocks " << numBlocks << std::endl;
	std::cout << "Occupancy: " << (double)activeWarps / maxWarps * 100 << "%" << std::endl;

	clock_t start = clock();
	for (int i = 0; i < iter; i++) {

		int task_size = NUM_BLOCK * NUM_THREAD;
		updateNormalKernel << < NUM_BLOCK, NUM_THREAD >> > (d_all_face_normal, d_all_centriod,
			d_all_face_area, d_filtered_normal, temp_filtered, num, sigma_s, sigma_r);
		cudaDeviceSynchronize();
		checkCudaErrors(cudaMemcpy(d_filtered_normal, temp_filtered, size, cudaMemcpyDeviceToDevice));
	}
	clock_t end = clock();
	std::cout << "kernel is call" << std::endl;
	std::cout << "kernrl invoke average time is " << (static_cast<double>(end - start ) / CLOCKS_PER_SEC)  / iter << std::endl;


	//Allocate host memory to device
	size_t size_filtered = num * sizeof(GPURolling::Normal);
	*filtered_ptr = (GPURolling::Normal*)malloc(size_filtered);

	//Copy device memory to host
	checkCudaErrors(cudaMemcpy(*filtered_ptr, d_filtered_normal, size_filtered, cudaMemcpyDeviceToHost));

	//Free device global memory
	checkCudaErrors(cudaFree(temp_filtered));
	checkCudaErrors(cudaFree(d_all_centriod));
	checkCudaErrors(cudaFree(d_all_face_normal));
	checkCudaErrors(cudaFree(d_all_face_area));
}


extern "C" void
updateVertexGlobal(std::vector <GPURolling::Point> all_centriod, 
	std::vector<GPURolling::Point> all_vertex, std::vector<double> all_face_area, 
	std::vector<GPURolling::Normal> filtered_normal, 
	GPURolling::Point **new_points, double sigma_s, double sigma_r)
{
	std::cout << "gpu update verteices begin" << std::endl;
	size_t face_num = all_centriod.size();
	size_t vertex_num = all_vertex.size();

	//allocate device memory
	GPURolling::Point *d_all_centriod = NULL;
	GPURolling::Point *d_all_vertex = NULL;
	GPURolling::Normal *d_filtered_normal = NULL;
	GPURolling::Normal *d_new_points = NULL;
	double *d_all_face_area = NULL;
	checkCudaErrors(cudaMalloc((void**)&d_all_centriod, face_num * sizeof(GPURolling::Point)));
	checkCudaErrors(cudaMalloc((void**)&d_all_vertex, vertex_num * sizeof(GPURolling::Point)));
	checkCudaErrors(cudaMalloc((void**)&d_filtered_normal, face_num * sizeof(GPURolling::Normal)));
	checkCudaErrors(cudaMalloc((void**)&d_all_face_area, face_num * sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&d_new_points, vertex_num * sizeof(GPURolling::Point)));

	//copy memory form host to device
	checkCudaErrors(cudaMemcpy(d_all_centriod, &all_centriod[0], face_num * sizeof(GPURolling::Point), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_all_vertex, &all_vertex[0], vertex_num * sizeof(GPURolling::Point), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_filtered_normal, &filtered_normal[0], face_num * sizeof(GPURolling::Point), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_all_face_area, &all_face_area[0], face_num * sizeof(double), cudaMemcpyHostToDevice));

	std::cout << "update vertex kernel begin " << std::endl;
	//kernel
	updateVertexKernel << < NUM_BLOCK, NUM_THREAD >> > (d_all_centriod, d_all_vertex, d_filtered_normal,
		d_all_face_area, d_new_points,
		face_num, vertex_num, sigma_s, sigma_r);

	//Allocate host memory
	assert(new_points != NULL);
	*new_points = (GPURolling::Point*)malloc(vertex_num * sizeof(GPURolling::Point));
	checkCudaErrors(cudaMemcpy(*new_points, d_new_points, sizeof(GPURolling::Point) * vertex_num, cudaMemcpyDeviceToHost));

	std::cout << "update vertex kernel end " << std::endl;
	//Free
	checkCudaErrors(cudaFree(d_all_centriod));
	checkCudaErrors(cudaFree(d_all_face_area));
	checkCudaErrors(cudaFree(d_all_vertex));
	checkCudaErrors(cudaFree(d_filtered_normal));
}

__global__ void
updateNormalKernel(const GPURolling::Normal *dnormal, const GPURolling::Point *dpoint,
	const double *darea, 		/*parameter*/
	const GPURolling::Normal *dfiltered, GPURolling::Normal *temp,
	const unsigned int num, const double sigma_s, const double sigma_r)
{
	const int tid = blockDim.x * blockIdx.x + threadIdx.x;

	unsigned int i = tid;
	while (i < num) 
	{
		GPURolling::Normal ni = dnormal[i];							/*read only*/
		GPURolling::Normal ni_k = dfiltered[i];						/*read and write*/
		GPURolling::Point ci = dpoint[i];							/*read only*/
		GPURolling::Normal nt = GPURolling::Normal(0.0, 0.0, 0.0);
		//guaasian filtering
		for (int j = 0; j < num; j++)
		{
			GPURolling::Normal nj_k = dfiltered[j];
			GPURolling::Point cj = dpoint[j];
			GPURolling::Normal nj = dnormal[j];
			double spatial_weight = dGaussian(sigma_s, (ci - cj).len());
			double range_weight = dGaussian(sigma_r, (ni_k - nj_k).len());
			double factor = darea[j] * spatial_weight *  range_weight;
			nt += nj *  factor;
		}
		nt.normalize();
		__syncthreads();

		//synchronize threads
		temp[i] = nt;
		i += STRIDE;
	}
}

__global__ void 
updateVertexKernel(const GPURolling::Point *d_all_centroid, const GPURolling::Point *d_all_vertex, const GPURolling::Normal *d_filtered_normal,
	const double *d_all_area, GPURolling::Point *new_points,
	const unsigned int face_num, const unsigned int vertex_num,
	double sigma_s, double sigma_r)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int i = tid;
	while (i < vertex_num)
	{
		//GPURolling::Point temp_centroid = d_all_centroid[i];
		GPURolling::Point pi = d_all_vertex[i];
		GPURolling::Point temp_point = GPURolling::Point(0.0, 0.0, 0.0);
		double total_weight = 0.0;
		for (int j = 0; j < face_num; j++)
		{
			GPURolling::Point cj = d_all_centroid[j];
			GPURolling::Normal nj = d_filtered_normal[j];
			double temp_area = d_all_area[j];
			double spatial_distance = (pi - cj).len();
			double range_distance = (nj * (nj | (pi - cj))).len();
			double spatial_weight = dGaussian(sigma_s, spatial_distance);
			double range_weight = dGaussian(sigma_r, range_distance);
			double weight = temp_area * spatial_weight * range_weight;
			temp_point += nj * ((nj | (cj - pi)) * weight);
			total_weight += weight;
		}
		temp_point /= total_weight;
		__syncthreads();

		new_points[i] = pi + temp_point;
		i += STRIDE;
	}
}

#endif _GPUROLLING_CUH