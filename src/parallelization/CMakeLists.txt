MESSAGE (STATUS "${CMAKE_CURRENT_SOURCE_DIR}:kai begin ==================================")

include ("${ClickHouse_SOURCE_DIR}/cmake/dbms_glob_sources.cmake")
#################################### for cuda ####################################
# 把common_path下的所有.cu文件 加入prefix_cuda变量
macro(add_cuda_headers_and_sources prefix common_path)
    add_glob (${prefix}_headers ${common_path}/*.cuh ${common_path}/*.h)
    add_glob (${prefix}_sources ${common_path}/*.cu ${common_path}/*.cuh ${common_path}/*.cpp ${common_path}/*.h)
endmacro()
#
set(cuda_include_dir /usr/local/cuda/include)
set(cuda_lib_dir /usr/local/cuda/lib64)
#
include_directories(${cuda_include_dir})
link_directories(${cuda_lib_dir})
#
add_cuda_headers_and_sources (clickhouse_aggregate_functions_cuda .)
add_library (clickhouse_aggregate_functions_cuda SHARED ${clickhouse_aggregate_functions_cuda_sources})    # ${clickhouse_aggregate_functions_cuda_headers}
#
set_target_properties(clickhouse_aggregate_functions_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(clickhouse_aggregate_functions_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(clickhouse_aggregate_functions_cuda PUBLIC cuda_std_17)

#target_include_directories(clickhouse_aggregate_functions_cuda PUBLIC ${cuda_include_dir})
target_link_libraries(clickhouse_aggregate_functions_cuda PUBLIC ${cuda_lib_dir}/libcudart_static.a)
#################################### for cuda ####################################

MESSAGE (STATUS "${CMAKE_CURRENT_SOURCE_DIR}:kai end ==================================")