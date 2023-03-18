#pragma once

using AggregateDataPtr = char *;

template <typename T>
void addBatchSumCuda(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long if_argument_pos);

extern template void addBatchSumCuda<signed char>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<short>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<int>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<long>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);

extern template void addBatchSumCuda<unsigned char>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<unsigned short>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<unsigned int>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<unsigned long>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);

extern template void addBatchSumCuda<float>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);
extern template void addBatchSumCuda<double>(size_t row_nums, AggregateDataPtr __restrict place, const char * start_row, long);

/*template <typename T>
void addBatchSumCuda(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long if_argument_pos);

extern template void addBatchSumCuda<signed char>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<short>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<int>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<long>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);

extern template void addBatchSumCuda<unsigned char>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<unsigned short>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<unsigned int>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<unsigned long>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);

extern template void addBatchSumCuda<float>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);
extern template void addBatchSumCuda<double>(size_t row_nums, AggregateDataPtr * places, size_t place_offset, const char * start_row, long);*/
