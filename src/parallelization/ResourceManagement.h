#pragma once

#include <mutex>
#include <list>
#include <stdexcept>
#include <memory>
#include <unordered_map>
#include <iostream>
#include "DeviceManagement.cuh"

class ThreadSafeQueue{
private:
    std::list<size_t> kai_queue;
    std::mutex kai_mutex;

public:
    ThreadSafeQueue()=default;

    ~ThreadSafeQueue()=default;

    void push_s(const size_t & val){
        kai_mutex.lock();
        kai_queue.push_back(val);
        kai_mutex.unlock();
    }

    size_t pop_s(){
        kai_mutex.lock();

        if(kai_queue.empty()){
            kai_mutex.unlock();
            throw std::logic_error("The cuda resources are gone!");
        }
        size_t val = kai_queue.front();
        kai_queue.pop_front();

        kai_mutex.unlock();

        return val;
    }

    size_t getSize(){
        return kai_queue.size();
    }
};


class ResourceManagement{
private:
    ThreadSafeQueue resourceQueue;
    size_t num_threads = 12;
    bool resourceInitialized = false;

public:
    ResourceManagement() = default;

    ~ResourceManagement() = default;

    void setNumThreads(size_t _num_threads) {
        if (_num_threads < 1) {
            _num_threads = 1;
        }
        num_threads = _num_threads;
        callSetNumThreads(_num_threads);
    }

    void initResource(size_t sizeof_T, size_t sizeof_TResult, size_t num_rows, size_t _num_threads = 12){
        callInitMemSpace(sizeof_T, sizeof_TResult, num_rows);

        if(resourceInitialized) return;

        setNumThreads(_num_threads);
        for (size_t i = 0; i < num_threads; i++) {
            resourceQueue.push_s(i);
        }
        callInitDeviceStreams();

        resourceInitialized = true;
    }

    void checkResource(size_t resourceIdx){
        if(resourceIdx >= num_threads){
            throw std::out_of_range("The cuda resource idx is out of range!");
        }
    }

    size_t getResourceIdx() {
        size_t idx = resourceQueue.pop_s();
        return idx;
    }

    void returnResourceIdx(size_t resourceIdx) {
        checkResource(resourceIdx);
        resourceQueue.push_s(resourceIdx);
    }

    void **getAllMemPtrs(size_t resourceIdx) {
        checkResource(resourceIdx);
        return callGetAllMemPtrs(resourceIdx);
    }


    void * getStream(size_t resourceIdx) {
        checkResource(resourceIdx);
        return callGetStream(resourceIdx);
    }
};

class ResourceManagementOnce : public ResourceManagement {

    static std::unique_ptr<ResourceManagementOnce> the_instance;

    ResourceManagementOnce()=default;

public:
    static void initialize() {
        if (the_instance) {
            return;
        }
        the_instance.reset(new ResourceManagementOnce());
    }

    static ResourceManagementOnce & instance() {
        if (!the_instance) {
            initialize();
        }
        return *the_instance;
    }
};




