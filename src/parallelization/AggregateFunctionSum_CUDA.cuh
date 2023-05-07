#pragma once

template <typename T, typename TResult>
void addBatchSumCuda(size_t, TResult *, const char *, long, void* , void**);
