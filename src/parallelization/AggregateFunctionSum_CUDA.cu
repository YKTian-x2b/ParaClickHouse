#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "AggregateFunctionSum_CUDA.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BDIMX_Y 256
#define UNROLL_NUM 8

#define CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) \
        { \
            printf("Error: %s: %d, ", __FILE__, __LINE__); \
            printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    }

// 函数需要考虑溢出
template <typename T>
__global__ void addBatchSumKernelV1(T * input, T * output, size_t stride, size_t n, T * input_back)
{
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * stride + threadIdx.x;
    T * data_in = input + blockIdx.x * blockDim.x * stride;
    //
    T * data_in_back = input_back + blockIdx.x * blockDim.x * stride;
    std::printf("nihao\n");

    // 这里的循环展开代码 应该随着 stride或UNROLL_NUM 的值而改变
    if (7 * blockDim.x + idx < n)
    {
        T a1 = data_in[1 * blockDim.x + tid];
        T a2 = data_in[2 * blockDim.x + tid];
        T a3 = data_in[3 * blockDim.x + tid];
        T a4 = data_in[4 * blockDim.x + tid];
        T a5 = data_in[5 * blockDim.x + tid];
        T a6 = data_in[6 * blockDim.x + tid];
        T a7 = data_in[7 * blockDim.x + tid];
        data_in[tid] += a1 + a2 + a3 + a4 + a5 + a6 + a7;
        //
        data_in_back[tid] = data_in[tid];
        data_in_back[1 * blockDim.x + tid] = a1;
        data_in_back[2 * blockDim.x + tid] = a2;
        data_in_back[3 * blockDim.x + tid] = a3;
        data_in_back[4 * blockDim.x + tid] = a4;
        data_in_back[5 * blockDim.x + tid] = a5;
        data_in_back[6 * blockDim.x + tid] = a6;
        data_in_back[7 * blockDim.x + tid] = a7;

    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512)
    {
        data_in[tid] += data_in[tid + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256)
    {
        data_in[tid] += data_in[tid + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128)
    {
        data_in[tid] += data_in[tid + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64)
    {
        data_in[tid] += data_in[tid + 64];
    }
    __syncthreads();

    if (tid < 32)
    {
        //
        data_in_back[tid] = data_in[tid];
        //
        volatile T * v_s_mem = data_in;
        v_s_mem[tid] += v_s_mem[tid + 32];
        v_s_mem[tid] += v_s_mem[tid + 16];
        v_s_mem[tid] += v_s_mem[tid + 8];
        v_s_mem[tid] += v_s_mem[tid + 4];
        v_s_mem[tid] += v_s_mem[tid + 2];
        v_s_mem[tid] += v_s_mem[tid + 1];
    }
    if (tid == 0)
    {
        output[bid] = data_in[0];
    }
}

__global__ void sayHello(){
    std::printf("Hello");
}

//
template <typename T>
void addBatchSumCuda(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long if_argument_pos)
{
    cudaSetDevice(0);
    /*dim3 block_size(BDIMX_Y);
    size_t block_level_size = block_size.x * UNROLL_NUM;
    dim3 grid_size((row_nums - 1) / block_level_size + 1);
    printf("block_size: %d, grid_size: %d, row_nums: %d\n", block_size.x, grid_size.x, row_nums);
    if (if_argument_pos >= 0)
    {
        // do something
        printf("if_argument_pos = -1\n");
    }
    else
    {
        // CPU && GPU Mem Alloc
        size_t col_vec_bytes = row_nums * sizeof(T);
        size_t padded_vec_bytes = block_size.x * UNROLL_NUM * grid_size.x * sizeof(T);
        T * d_input;
        cudaMalloc((T **)&d_input, padded_vec_bytes);
        cudaMemset(d_input, 0, padded_vec_bytes);
        // CHECK(cudaMalloc(&d_input, padded_vec_bytes));
        printf("=========== inp ================\n");
        T * d_output, * h_output;
        size_t res_bytes = sizeof(T) * grid_size.x;
        cudaMalloc((T **)&d_output, res_bytes);
        // CHECK(cudaMalloc(&d_output, res_bytes));
        h_output = (T *)std::malloc(res_bytes);
        printf("=========== outp ================\n");
        // Data transfer from CPU Mem to GPU Mem
        // cudaMemcpy(d_input, start_row, col_vec_bytes, cudaMemcpyHostToDevice);
        // CHECK(cudaMemcpy(d_input, &start_row, col_vec_bytes, cudaMemcpyHostToDevice));
        printf("=========== cudaMemcpy inp ================\n");
        // kernel
        T * d_input_back, * h_input_back;
        cudaMalloc((T **)&d_input_back, padded_vec_bytes);
        h_input_back = (T *)std::malloc(padded_vec_bytes);
        memset(h_input_back, -1, padded_vec_bytes);
        addBatchSumKernelV1<T><<<grid_size, block_size>>>(d_input, d_output, UNROLL_NUM, row_nums, d_input_back);
        cudaDeviceSynchronize();
        cudaMemcpy(h_input_back, d_input_back, padded_vec_bytes, cudaMemcpyDeviceToHost);
        for(int i = 0; i < 10; i++){
            printf("d_input_back_%d: %d\n", i, h_input_back[i]);
        }
        printf("=========== addBatchSumKernelV1 ================\n");
        // Data transfer from GPU Mem to CPU Mem
        cudaMemcpy(h_output, d_output, res_bytes, cudaMemcpyDeviceToHost);
        // CHECK(cudaMemcpy(h_output, d_output, res_bytes, cudaMemcpyDeviceToHost));
        printf("=========== cudaMemcpy outp ================\n");

        T res = 0;
        for (int i = 0; i < grid_size.x; i++)
        {
            res += h_output[i];
            printf("h_output_%d: %d\n", i, h_output[i]);
        }
        printf("res: %d\n", res);

        // Dealloc CPU && GPU Mem
        cudaFree(d_input);
        cudaFree(d_output);
//        CHECK(cudaFree(d_input));
//        CHECK(cudaFree(d_output));
        std::free(h_output);

        cudaFree(d_input_back);
        std::free(h_input_back);
    }*/
    printf("before %s\n", cudaGetErrorString(cudaGetLastError()));
    sayHello<<<1,1>>>();
    cudaDeviceSynchronize();
    printf("after %s\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceReset();
}

template void addBatchSumCuda<signed char>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<short>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<int>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<long>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);

template void addBatchSumCuda<unsigned char>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<unsigned short>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<unsigned int>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<unsigned long>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);

template void addBatchSumCuda<float>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
template void addBatchSumCuda<double>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);

/*template <typename T>
void addBatchSumCuda(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long if_argument_pos){
    // Poco::Logger * log = &Poco::Logger::get("AggregateFunctionSum_CUDA");

    cudaSetDevice(0);
    dim3 block_size(BDIMX_Y);
    size_t block_level_size = block_size.x * UNROLL_NUM;
    dim3 grid_size((row_nums - 1)/block_level_size + 1);

    if(if_argument_pos >= 0){
        // do something
        std::cout << "if_argument_pos = -1" << std::endl;
    }
    else{
        // CPU && GPU Mem Alloc
        size_t col_vec_bytes = row_nums * sizeof(T);
        T * d_input;
        CHECK(cudaMalloc((T**)&d_input, col_vec_bytes));

        T * d_output, *h_output;
        size_t res_bytes = sizeof(T) * grid_size.x;
        CHECK(cudaMalloc(&d_output, res_bytes));
        h_output = (T *)std::malloc(res_bytes);

        // Data transfer from CPU Mem to GPU Mem
         CHECK(cudaMemcpy(d_input, &start_row, col_vec_bytes, cudaMemcpyHostToDevice));
        // kernel
        addBatchSumKernelV1<T><<<grid_size, block_size>>>(d_input, d_output, UNROLL_NUM, row_nums);
        // Data transfer from GPU Mem to CPU Mem
        CHECK(cudaMemcpy(h_output, d_output, res_bytes, cudaMemcpyDeviceToHost));

        T res = 0;
        for(int i = 0; i < grid_size.x; i++){
            res += h_output[i];
        }
        std::cout << "res: " << res << std::endl;

        // LOG_INFO(log, "cuda_sum res = 114514, ======================================================================================");

        // Dealloc CPU && GPU Mem
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        free(h_output);
    }

    cudaDeviceReset();
}

template void addBatchSumCuda<signed char>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<short>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<int>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<long>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);

template void addBatchSumCuda<unsigned char>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<unsigned short>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<unsigned int>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<unsigned long>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);

template void addBatchSumCuda<float>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
template void addBatchSumCuda<double>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);*/
