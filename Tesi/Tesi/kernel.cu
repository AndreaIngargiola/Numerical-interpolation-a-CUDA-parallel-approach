#include "hpc.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define MAXTHREADINBLOCK 1024
#define MAXSIDELENGHT 32
double* t;
double* z;

__global__ void find_t_z(dim3 blockDim, double* c_x, double* t, double* z, int n) {

    // Compute absolute thread id, finding the right indexes for array access.
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int arrayId = j + i * n;
    
    if (arrayId < n * n && j<n && i<n && j!=i) {
        t[arrayId] = c_x[j] - c_x[i];       // Compute xj - xi and save the result in t[i][j] (treated as a flatten 2D array)
        z[arrayId] = c_x[i]/ t[arrayId];    // Compute xi/(xj - xi) and save the result in z[i][j] (treated as a flatten 2D array)
    }
}


// Helper function for using CUDA to pre-calculate the domain-indipendent part of the Lagrange interpolation polynomial.
cudaError_t l_compute_constants(int n, double* x, double* y)
{
    cudaError_t cudaStatus;
    double *c_x, *c_y;
    int *c_n;
    dim3 blocks;
    dim3 threads;
    n = n + 1; //now n is the number of nodes and not the rank of the polynomial.

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Prepare device memory.
    cudaSafeCall(cudaMalloc((void**)&c_x, n*sizeof(double)));
    cudaSafeCall(cudaMalloc((void**)&c_y, n*sizeof(double)));

    // Copy host variables in device.
    cudaSafeCall(cudaMemcpy(c_x, x, n*sizeof(double), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(c_y, y, n*sizeof(double), cudaMemcpyHostToDevice));

    // Prepare auxiliary data structures.
    cudaSafeCall(cudaMalloc((void**)&t, n * n * sizeof(double)));
    cudaSafeCall(cudaMalloc((void**)&z,n * n * sizeof(double)));

    // Compute grid dimension
    threads = dim3(32, 32, 1);
    int prov = (n / MAXSIDELENGHT) + 1;
    blocks = dim3(prov, prov, 1);

    // Compute all possible constants and store the results in t and z array.
    find_t_z <<<blocks, threads>>> (threads, c_x, t, z, n);

   // cudaDeviceSynchronize waits for the kernel to finish, and returns
   // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching find_t_v kernel!\n", cudaStatus);
        goto Error;
    }

    // If everything is ok, the polynomial constant are computed and memorized in device's global memory.
    cudaFree(c_x);
    cudaFree(c_y);

Error:
    return cudaStatus;
}


__global__ void intra_block_computation(dim3 blockSize, double *y, int n, double* t, double* z, double* domain, double *codomain)
{
    //grid indexing variables
    unsigned int block_global_Id = blockIdx.x + blockIdx.y * gridDim.x;
    int j = threadIdx.x + blockIdx.y * blockDim.x;
    int i = threadIdx.y;
    
    int global_tid;
    int tid;
    
    if (j < n) {
        tid = threadIdx.x + (i * blockDim.x);
        global_tid = j + (i * n);
        
    }
    else {
        tid = -1;
        global_tid = -1;
    }
    
    unsigned int index_2d = j + (i * n);              //a 2d index disconnected from the tid, for referencing bigger domains.

    //shared memory variables
    __shared__ double xs;

    extern __shared__ float l[];
    if(tid != -1)l[tid] = 1;            //initialize l[tid] to the multiplication neutral element: this is needed for threads that wants to read out of bound elements or for threads with i == j.

    double gamma;
    
    if (tid == 0) {
        xs = domain[blockIdx.x];
    }
    __syncthreads();
    
    //find all gammas.
    while (i < n) {
        if (j < n) {
            index_2d = j + (i * n);
            if (i != j) {
                //coalesced access to global memory for point a (is always in-bound thanks to the while condition)
                gamma = (xs * ((double)1 / t[index_2d])) - z[index_2d];
            }
            else {
                gamma = 1;
            }
            l[tid] *= gamma;
        }
        i += blockSize.y;
    }
    i = threadIdx.y;

    __syncthreads();

    //reduce the l array columns (only between columns) like a binary tree to find definitive n Lj 
    if (j < n) {
        if (i+16< blockDim.y && i < 16){ l[tid] *= l[tid +  blockDim.x * 16]; }  __syncthreads(); 
        if (i+8 < blockDim.y && i < 8) { l[tid] *= l[tid +  blockDim.x * 8]; }  __syncthreads(); 
        if (i+4 < blockDim.y && i < 4) { l[tid] *= l[tid +  blockDim.x * 4]; }  __syncthreads(); 
        if (i+2 < blockDim.y && i < 2) { l[tid] *= l[tid +  blockDim.x * 2]; }  __syncthreads(); 
        if (i+1 < blockDim.y && i < 1) { l[tid] *= l[tid +  blockDim.x * 1]; }  __syncthreads(); 
    }
    __syncthreads();
   
    // Now that the real Lj are saved in the first n places of the shared array "l", all the partial results in the array must be transformed in the neutral summation element.
    if (i > 0 && j < n) {
        l[tid] = 0;
    }
    else if(global_tid < n && global_tid != -1){
        l[tid] = l[tid] * y[global_tid];  // Meanwhile, all the real Lj(xs) must be multiplied by Yj.
    }
    __syncthreads();

    if (j < n && i == 0) {
        //reduce the l array like a binary tree
        if (threadIdx.x + 16< blockDim.x && j + 16< n && threadIdx.x < 16){ l[tid] += l[tid + 16];}  __syncthreads();
        if (threadIdx.x + 8 < blockDim.x && j + 8 < n && threadIdx.x < 8) { l[tid] += l[tid + 8]; }  __syncthreads();
        if (threadIdx.x + 4 < blockDim.x && j + 4 < n && threadIdx.x < 4) { l[tid] += l[tid + 4]; }  __syncthreads();
        if (threadIdx.x + 2 < blockDim.x && j + 2 < n && threadIdx.x < 2) { l[tid] += l[tid + 2]; }  __syncthreads();
        if (threadIdx.x + 1 < blockDim.x && j + 1 < n && threadIdx.x < 1) { l[tid] += l[tid + 1]; }  __syncthreads();
        
        __syncthreads();

        //save the final result.
        if (threadIdx.x == 0 && threadIdx.y == 0) { 
            codomain[block_global_Id] = l[tid];
        }
    }
}

__global__ void inter_block_reduction(double* ys, int res_size, int blocks_per_ys, double* res) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k;
    double alpha = 0;

    if (tid < res_size) {
        for (k = 0; k < blocks_per_ys; k++) {
            alpha = alpha + ys[tid + (k * res_size)];
        }
        res[tid] = alpha;
    }
}

double* l_compute_codomain(int n, double* x, double* y, double *xs, double *ys, int res_size)
{
    cudaError_t cudaStatus;
    dim3 threads, blocks;
    int sharedMemory;
    double *domain, *codomain, *codomain_to_blockreduce;
    double *c_y;
    n = n + 1; //now n is the number of nodes and not the rank of the polynomial.
               //res_size is the number S of points that must be interpolated.

    if (!t || !z) {
        fprintf(stderr, "Constant arrays not found! Call compute_lagrange() before attempting to use it.");
        goto Error;
    }

    // Compute grid dimension
    int blocks_per_ys = (n / MAXSIDELENGHT)+1;

    blocks = dim3(res_size, blocks_per_ys, 1);
    
    if (n < MAXSIDELENGHT) {
        threads = dim3(n, n, 1);
        sharedMemory = ((n * n) + n + 1) * sizeof(double);
    }
    else {
        threads = dim3(MAXSIDELENGHT, MAXSIDELENGHT, 1);
        sharedMemory = ((MAXSIDELENGHT * MAXSIDELENGHT) + n + 1) * sizeof(double);
    }
    ys = (double*)malloc(res_size * sizeof(double));

    cudaSafeCall(cudaMalloc((void**)&domain, res_size * sizeof(double)));
    cudaSafeCall(cudaMalloc((void**)&codomain, res_size * sizeof(double)));
    cudaSafeCall(cudaMalloc((void**)&codomain_to_blockreduce, res_size * blocks_per_ys * sizeof(double)));

    cudaSafeCall(cudaMemcpy(domain, xs, res_size * sizeof(double), cudaMemcpyHostToDevice));

    cudaSafeCall(cudaMalloc((void**)&c_y, n * sizeof(double)));
    cudaSafeCall(cudaMemcpy(c_y, y, n * sizeof(double), cudaMemcpyHostToDevice));

    intra_block_computation <<<blocks, threads, sharedMemory >>> (threads, c_y, n, t, z, domain, codomain_to_blockreduce);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess /*  && count==0  */) {
        fprintf(stderr, "error in intra_block_computation kernel: %s \n", cudaGetErrorString(error));
        goto Error;
    }
  
    cudaStatus = cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess /*  && count==0  */) {
        fprintf(stderr, "error in intra_block_computation kernel after cudaDeviceSynchronize: %s \n", cudaGetErrorString(error));
        goto Error;
    }

    threads = dim3(1024, 1, 1);
    blocks = dim3((res_size / 1024) + 1, 1, 1);

    inter_block_reduction<<<blocks, threads>>>(codomain_to_blockreduce, res_size, blocks_per_ys, codomain);
    error = cudaGetLastError();
    if (error != cudaSuccess /*  && count==0  */) {
        fprintf(stderr, "error in inter_block_reduction kernel: %s \n", cudaGetErrorString(error));
        goto Error;
    }
  
    cudaStatus = cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess /*  && count==0  */) {
        fprintf(stderr, "error in inter_block_reduction kernel after cudaDeviceSynchronize: %s \n", cudaGetErrorString(error));
        goto Error;
    }

    cudaSafeCall(cudaMemcpy(ys, codomain, res_size * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(domain);
    cudaFree(codomain);
    cudaFree(codomain_to_blockreduce);
    cudaFree(c_y);
    cudaDeviceReset();

    return ys;
Error:
    return NULL;
}

double* l_compute_lagrange(int n, double* x, double* y, double* xs, double* ys, int res_size) {
    double* result;
    n = n + 1; //now n is the number of nodes and not the rank of the polynomial.
    cudaError_t cudaStatus = l_compute_constants(n, x, y);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "l_compute_constants failed!");
        return NULL;
    }
    result = l_compute_codomain(n, x, y, xs, ys, res_size);
    return result;
}