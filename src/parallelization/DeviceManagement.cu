#include "DeviceManagement.cuh"
#include "Common_CUDA.h"

class DeviceManagement {
private:
    struct MemSpace{
        void * d_input;
        void * d_result;
        void * h_result;
    };

    MemSpace * mem_s = nullptr;
    cudaStream_t * streams = nullptr;

    size_t num_threads = 12;

    bool streamsInitialized = false;
    bool memSpaceInitialized = false;
    size_t lastTSize = 0;
    size_t lastTResultSize = 0;

    void *** allThreadsAllMemPtrs = nullptr;
public:
    const size_t d_block_size = 256;
    const size_t unroll_num = 8;

    DeviceManagement() = default;

    ~DeviceManagement() {
        for (size_t i = 0; i < num_threads; i++) {
            CHECK(cudaStreamDestroy(streams[i]));
            CHECK(cudaFree(mem_s[i].d_input));
            CHECK(cudaFree(mem_s[i].d_result));
            free(mem_s[i].h_result);
            free(allThreadsAllMemPtrs[i]);
        }
        free(streams);

        free(mem_s);

        free(allThreadsAllMemPtrs);
        printf("~DeviceManagement();\n");
    }

    void freeOldMemSpace(){
        for (size_t i = 0; i < num_threads; i++) {
            CHECK(cudaFree(mem_s[i].d_input));
            CHECK(cudaFree(mem_s[i].d_result));
            free(mem_s[i].h_result);
            free(allThreadsAllMemPtrs[i]);
        }
        free(mem_s);
        free(allThreadsAllMemPtrs);
    }

    void setNumThreads(size_t _num_threads) {
        if (_num_threads < 1) {
            _num_threads = 1;
        }
        num_threads = _num_threads;
    }

    void initDeviceStreams() {
        if (streamsInitialized) return;

        streams = (cudaStream_t *)malloc(num_threads * sizeof(cudaStream_t));
        for (int i = 0; i < num_threads; i++) {
            CHECK(cudaStreamCreate(&streams[i]));
        }
        streamsInitialized = true;
    }

    void initAggregateFunctionSumMemSpace(size_t sizeof_T, size_t sizeof_TResult, size_t num_rows) {
        if (memSpaceInitialized && lastTSize == sizeof_T && lastTResultSize == sizeof_TResult){
            return;
        }

        if(lastTSize && lastTResultSize){
            freeOldMemSpace();
        }

        mem_s = (MemSpace *)malloc(sizeof(MemSpace) * num_threads);

        size_t grid_size = (num_rows - 1) / (d_block_size * unroll_num) + 1;
        size_t d_padded_input_bytes = d_block_size * unroll_num * grid_size * sizeof_T;
        size_t res_bytes = grid_size * sizeof_TResult;

        allThreadsAllMemPtrs = (void***)malloc(sizeof(void*)*num_threads);

        for (size_t i = 0; i < num_threads; i++) {
            CHECK(cudaMalloc(&mem_s[i].d_input, d_padded_input_bytes));
            CHECK(cudaMalloc(&mem_s[i].d_result, res_bytes));
            mem_s[i].h_result = malloc(res_bytes);

            allThreadsAllMemPtrs[i] = (void**)malloc(sizeof(void*)*4);
            allThreadsAllMemPtrs[i][0] = mem_s[i].d_input;
            allThreadsAllMemPtrs[i][1] = mem_s[i].d_result;
            allThreadsAllMemPtrs[i][2] = mem_s[i].h_result;
        }

        lastTSize = sizeof_T;
        lastTResultSize = sizeof_TResult;
        memSpaceInitialized = true;
    }

    void** getAllMemPtrs(size_t resourceIdx){
        return allThreadsAllMemPtrs[resourceIdx];
    }

    cudaStream_t * getStream(size_t resourceIdx) {
        return &streams[resourceIdx];
    }
};

class DeviceManagementOnce : public DeviceManagement {

    static DeviceManagementOnce * the_instance;

    DeviceManagementOnce() = default;

public:
    static void initialize() {
        if (the_instance) {
            return;
        }
        the_instance = new DeviceManagementOnce();
    }

    static DeviceManagementOnce &instance() {
        if (!the_instance) {
            initialize();
        }
        return *the_instance;
    }
};

DeviceManagementOnce * DeviceManagementOnce::the_instance;


void callInitDeviceStreams() {
    DeviceManagementOnce::instance().initDeviceStreams();
}

void callInitMemSpace(size_t sizeof_T, size_t sizeof_TResult, size_t num_rows) {
    DeviceManagementOnce::instance().initAggregateFunctionSumMemSpace(sizeof_T, sizeof_TResult, num_rows);
}

size_t getDBlockSize() {
    return DeviceManagementOnce::instance().d_block_size;
}

size_t getUnrollNum() {
    return DeviceManagementOnce::instance().unroll_num;
}

void **callGetAllMemPtrs(size_t resourceIdx) {
    return DeviceManagementOnce::instance().getAllMemPtrs(resourceIdx);
}

void * callGetStream(size_t resourceIdx) {
    return DeviceManagementOnce::instance().getStream(resourceIdx);
}

void callSetNumThreads(size_t num_threads){
    return DeviceManagementOnce::instance().setNumThreads(num_threads);
}