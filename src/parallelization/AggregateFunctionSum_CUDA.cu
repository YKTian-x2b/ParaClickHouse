#include <cstdlib>
#include <cstring>

#include "Common_CUDA.h"
#include "AggregateFunctionSum_CUDA.cuh"
#include "DeviceManagement.cuh"

#define BDIM 256
/*template<typename T, typename TResult>
__global__ void addBatchSumKernelV1(T *input, TResult *output, TResult *result, size_t stride, size_t n) {
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * stride + threadIdx.x;
    T *data_in = input + blockIdx.x * blockDim.x * stride;
    TResult *data_out = output + blockIdx.x * blockDim.x;

    // 这里的循环展开代码 应该随着 stride或UNROLL_NUM 的值而改变
    if (7 * blockDim.x + idx < n) {
        T a0 = data_in[tid];
        T a1 = data_in[1 * blockDim.x + tid];
        T a2 = data_in[2 * blockDim.x + tid];
        T a3 = data_in[3 * blockDim.x + tid];
        T a4 = data_in[4 * blockDim.x + tid];
        T a5 = data_in[5 * blockDim.x + tid];
        T a6 = data_in[6 * blockDim.x + tid];
        T a7 = data_in[7 * blockDim.x + tid];

        data_out[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) {
        data_out[tid] += data_out[tid + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        data_out[tid] += data_out[tid + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        data_out[tid] += data_out[tid + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        data_out[tid] += data_out[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        volatile TResult *v_s_mem = data_out;
        v_s_mem[tid] += v_s_mem[tid + 32];
        v_s_mem[tid] += v_s_mem[tid + 16];
        v_s_mem[tid] += v_s_mem[tid + 8];
        v_s_mem[tid] += v_s_mem[tid + 4];
        v_s_mem[tid] += v_s_mem[tid + 2];
        v_s_mem[tid] += v_s_mem[tid + 1];
    }
    if (tid == 0) {
        result[bid] = data_out[0];
    }
}*/

template<typename T, typename TResult>
__global__ void addBatchSumKernelV2(T *input, TResult *result, size_t stride, size_t n) {
    size_t bid = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x * stride + threadIdx.x;
    T *data_in = input + blockIdx.x * blockDim.x * stride;
    __shared__ TResult data_out[BDIM];

    // 这里的循环展开代码 应该随着 stride或UNROLL_NUM 的值而改变
    if (7 * blockDim.x + idx < n) {
        T a0 = data_in[tid];
        T a1 = data_in[1 * blockDim.x + tid];
        T a2 = data_in[2 * blockDim.x + tid];
        T a3 = data_in[3 * blockDim.x + tid];
        T a4 = data_in[4 * blockDim.x + tid];
        T a5 = data_in[5 * blockDim.x + tid];
        T a6 = data_in[6 * blockDim.x + tid];
        T a7 = data_in[7 * blockDim.x + tid];

        data_out[tid] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512) {
        data_out[tid] += data_out[tid + 512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        data_out[tid] += data_out[tid + 256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        data_out[tid] += data_out[tid + 128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        data_out[tid] += data_out[tid + 64];
    }
    __syncthreads();

    if (tid < 32) {
        volatile TResult *v_s_mem = data_out;
        v_s_mem[tid] += v_s_mem[tid + 32];
        v_s_mem[tid] += v_s_mem[tid + 16];
        v_s_mem[tid] += v_s_mem[tid + 8];
        v_s_mem[tid] += v_s_mem[tid + 4];
        v_s_mem[tid] += v_s_mem[tid + 2];
        v_s_mem[tid] += v_s_mem[tid + 1];
    }
    if (tid == 0) {
        result[bid] = data_out[0];
    }
}

template<typename T, typename TResult>
__host__ void addBatchSumCuda(size_t num_rows, TResult *result, const char *start_row, long if_argument_pos, void *_stream, void **allMemPtr) {
    dim3 block_size(getDBlockSize());
    size_t unroll_num = getUnrollNum();
    size_t block_level_size = block_size.x * unroll_num;
    dim3 grid_size((num_rows - 1) / block_level_size + 1);
    cudaStream_t stream = *((cudaStream_t *) _stream);
    // printf("block_size: %d, grid_size: %d, row_nums: %zu\n", block_size.x, grid_size.x, num_rows);

    // CPU && GPU Mem Alloc
    size_t col_vec_bytes = num_rows * sizeof(T);
    size_t num_data_s = block_size.x * unroll_num * grid_size.x;

    T *d_input;
    d_input = (T *) allMemPtr[0];
    CHECK(cudaMemsetAsync(d_input, 0, num_data_s * sizeof(T), stream));

    TResult *d_result, *h_result;
    size_t res_bytes = sizeof(TResult) * grid_size.x;
    d_result = (TResult *) allMemPtr[1];
    h_result = (TResult *) allMemPtr[2];

    // Data transfer from CPU Mem to GPU Mem
    CHECK(cudaMemcpyAsync(d_input, start_row, col_vec_bytes, cudaMemcpyHostToDevice, stream));

    // kernel
    addBatchSumKernelV2<T, TResult><<<grid_size, block_size, 0, stream>>>(d_input, d_result, unroll_num, num_data_s);

    // Data transfer from GPU Mem to CPU Mem
    CHECK(cudaMemcpyAsync(h_result, d_result, res_bytes, cudaMemcpyDeviceToHost, stream));
    CHECK(cudaStreamSynchronize(stream));

    TResult res = 0;
    for (int i = 0; i < grid_size.x; i++) {
        res += h_result[i];
    }
    /*if constexpr (std::is_same_v<TResult, signed char>) {
        printf("res: %c\n", res);
    } else if constexpr (std::is_same_v<TResult, short>) {
        printf("res: %hd\n", res);
    } else if constexpr (std::is_same_v<TResult, int>) {
        printf("res: %d\n", res);
    } else if constexpr (std::is_same_v<TResult, long>) {
        printf("res: %ld\n", res);
    } else if constexpr (std::is_same_v<TResult, unsigned char>) {
        printf("res: %hhu\n", res);
    } else if constexpr (std::is_same_v<TResult, signed short>) {
        printf("res: %hu\n", res);
    } else if constexpr (std::is_same_v<TResult, unsigned>) {
        printf("res: %u\n", res);
    } else if constexpr (std::is_same_v<TResult, unsigned long>) {
        printf("res: %lu\n", res);
    } else if constexpr (std::is_same_v<TResult, float>) {
        printf("res: %f\n", res);
    } else if constexpr (std::is_same_v<TResult, double>) {
        printf("res: %lf\n", res);
    } else {
        printf("res: %d\n", res);
    }*/
    *result = res;

    // printf("addBatchSumCUDA end: %s\n", cudaGetErrorString(cudaGetLastError()));
}


template void addBatchSumCuda<signed char, signed char>(size_t, signed char *, const char  *, long, void *, void **);

template void addBatchSumCuda<short, short>(size_t, short *, const char  *, long, void *, void **);

template void addBatchSumCuda<int, int>(size_t, int *, const char  *, long, void *, void **);

template void addBatchSumCuda<long, long>(size_t, long *, const char *, long, void *, void **);

template void addBatchSumCuda<unsigned char, unsigned char>(size_t, unsigned char *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned short, unsigned short>(size_t, unsigned short *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned int, unsigned int>(size_t, unsigned int *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned long, unsigned long>(size_t, unsigned long *, const char  *, long, void *, void **);

template void addBatchSumCuda<signed char, long>(size_t, long *, const char  *, long, void *, void **);

template void addBatchSumCuda<short, long>(size_t, long *, const char  *, long, void *, void **);

template void addBatchSumCuda<int, long>(size_t, long *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned char, unsigned long>(size_t, unsigned long *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned short, unsigned long>(size_t, unsigned long *, const char *, long, void *, void **);

template void addBatchSumCuda<unsigned int, unsigned long>(size_t, unsigned long *,  const char *, long, void *, void **);

template void addBatchSumCuda<signed char, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<short, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<int, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<long, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned char, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned short, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned int, double>(size_t, double *, const char  *, long, void *, void **);

template void addBatchSumCuda<unsigned long, double>(size_t, double *, const char *, long, void *, void **);

template void addBatchSumCuda<float, float>(size_t, float *, const char *, long, void *, void **);

template void addBatchSumCuda<float, double>(size_t, double *, const char *, long, void *, void **);

template void addBatchSumCuda<double, double>(size_t, double *, const char *, long, void *, void **);


// #define BDIMX_Y 256
// #define UNROLL_NUM 8
/*
template <typename T, typename TResult>
__host__ void addBatchSumCuda(size_t num_rows, TResult * result, const char * start_row, long if_argument_pos, size_t)
{
    dim3 block_size(BDIMX_Y);
    size_t block_level_size = block_size.x * UNROLL_NUM;
    dim3 grid_size((num_rows - 1) / block_level_size + 1);
    // printf("block_size: %d, grid_size: %d, row_nums: %zu\n", block_size.x, grid_size.x, num_rows);

    if (if_argument_pos >= 0)
    {
        printf("if_argument_pos = -1\n");
    }
    else
    {
        // CPU && GPU Mem Alloc
        size_t col_vec_bytes = num_rows * sizeof(T);
        int num_data_s = block_size.x * UNROLL_NUM * grid_size.x;
        size_t padded_vec_bytes = num_data_s * sizeof(T);
        T * d_input;
        CHECK(cudaMalloc((T **)&d_input, padded_vec_bytes));
        CHECK(cudaMemset(d_input, 0, padded_vec_bytes));

        TResult * d_output, *d_result, * h_result;
        size_t d_output_bytes = block_size.x * grid_size.x * sizeof(TResult);
        size_t res_bytes = sizeof(TResult) * grid_size.x;
        CHECK(cudaMalloc((TResult **)&d_output, d_output_bytes));
        CHECK(cudaMalloc((TResult **)&d_result, res_bytes));
        h_result = (TResult *)std::malloc(res_bytes);

        // Data transfer from CPU Mem to GPU Mem
        CHECK(cudaMemcpy(d_input, start_row, col_vec_bytes, cudaMemcpyHostToDevice));

        // kernel
        // printf("addBatchSumKernelV1 before: %s\n", cudaGetErrorString(cudaGetLastError()));
        addBatchSumKernelV1<T, TResult><<<grid_size, block_size>>>(d_input, d_output, d_result, UNROLL_NUM, num_data_s);
        cudaDeviceSynchronize();

        // Data transfer from GPU Mem to CPU Mem
        CHECK(cudaMemcpy(h_result, d_result, res_bytes, cudaMemcpyDeviceToHost));

        TResult res = 0;
        for (int i = 0; i < grid_size.x; i++)
        {
            res += h_result[i];
        }
        if constexpr (std::is_same_v<TResult, signed char>) {
            printf("res: %c\n", res);
        }
        else if constexpr (std::is_same_v<TResult, short>) {
            printf("res: %hd\n", res);
        }
        else if constexpr (std::is_same_v<TResult, int>) {
            printf("res: %d\n", res);
        }
        else if constexpr (std::is_same_v<TResult, long>) {
            printf("res: %ld\n", res);
        }
        else if constexpr (std::is_same_v<TResult, unsigned char>) {
            printf("res: %hhu\n", res);
        }
        else if constexpr (std::is_same_v<TResult, signed short>) {
            printf("res: %hu\n", res);
        }
        else if constexpr (std::is_same_v<TResult, unsigned>) {
            printf("res: %u\n", res);
        }
        else if constexpr (std::is_same_v<TResult, unsigned long>) {
            printf("res: %lu\n", res);
        }
        else if constexpr (std::is_same_v<TResult, float>) {
            printf("res: %f\n", res);
        }
        else if constexpr (std::is_same_v<TResult, double>) {
            printf("res: %lf\n", res);
        }
        else {
            printf("res: %d\n", res);
        }
        *result = res;

        // Dealloc CPU && GPU Mem
        CHECK(cudaFree(d_input));
        CHECK(cudaFree(d_output));
        CHECK(cudaFree(d_result));
        free(h_result);
    }
    // printf("addBatchSumCUDA end: %s\n", cudaGetErrorString(cudaGetLastError()));
}*/
