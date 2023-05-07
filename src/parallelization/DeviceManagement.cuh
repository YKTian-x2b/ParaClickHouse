#pragma once

class DeviceManagement;

class DeviceManagementOnce;

extern void callInitDeviceStreams();

extern void callInitMemSpace(size_t sizeof_T, size_t sizeof_TResult, size_t num_rows);

extern size_t getDBlockSize();

extern size_t getUnrollNum();

extern void **callGetAllMemPtrs(size_t resourceIdx);

extern void * callGetStream(size_t resourceIdx);

extern void callSetNumThreads(size_t num_threads);


