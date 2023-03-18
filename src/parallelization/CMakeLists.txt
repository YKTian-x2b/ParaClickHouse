MESSAGE (STATUS "${CMAKE_CURRENT_SOURCE_DIR}:kai begin ==================================")

include ("${ClickHouse_SOURCE_DIR}/cmake/dbms_glob_sources.cmake")

#################################### for cuda ####################################
# 把common_path下的所有.cu文件 加入prefix_cuda变量
macro(add_cuda_headers_and_sources prefix common_path)
    add_glob (${prefix}_headers ${common_path}/*.cuh)
    add_glob (${prefix}_sources ${common_path}/*.cu ${common_path}/*.cuh)
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

target_include_directories(clickhouse_aggregate_functions_cuda PUBLIC ${cuda_include_dir})

target_link_libraries(clickhouse_aggregate_functions_cuda PUBLIC ${cuda_lib_dir}/libcudart.so)
# target_link_libraries (clickhouse_aggregate_functions_cuda PUBLIC dbms)
#################################### for cuda ####################################

#################################### for cpp ####################################
add_headers_and_sources(clickhouse_aggregate_functions_common ${CMAKE_CURRENT_SOURCE_DIR})
add_library(clickhouse_aggregate_functions_common SHARED ${clickhouse_aggregate_functions_common_sources})     # ${clickhouse_aggregate_functions_common_headers}

target_link_libraries(clickhouse_aggregate_functions_cuda PUBLIC "${cuda_lib_dir}/libcudart.so")
# target_link_libraries(clickhouse_aggregate_functions_common PUBLIC dbms)
#################################### for cpp ####################################

MESSAGE (STATUS "${CMAKE_CURRENT_SOURCE_DIR}:kai end ==================================")

#set(lib_for_cuda_headers)
#set(lib_for_cuda_sources)
#
# list(APPEND lib_for_cuda_headers ${PROJECT_SOURCE_DIR}/base/base/types.h)
#
#add_headers_and_sources(lib_for_cuda ${PROJECT_SOURCE_DIR}/src/Common)
#add_headers_and_sources(lib_for_cuda ${PROJECT_SOURCE_DIR}/src/Columns)
#add_headers_and_sources(lib_for_cuda ${PROJECT_SOURCE_DIR}/base/base)
#
# add_library(lib_for_cuda ${lib_for_cuda_sources} ${lib_for_cuda_headers})
#target_include_directories (lib_for_cuda PUBLIC
#        "${ClickHouse_SOURCE_DIR}/contrib" "${ClickHouse_BINARY_DIR}/contrib"
#        "${ClickHouse_SOURCE_DIR}/base" "${ClickHouse_BINARY_DIR}/base"
#        "${ClickHouse_SOURCE_DIR}/src" "${ClickHouse_BINARY_DIR}/src"
#        )
#target_include_directories (lib_for_cuda PUBLIC
#        "${ClickHouse_SOURCE_DIR}/base/base" "${ClickHouse_BINARY_DIR}/base/base")
#target_include_directories (clickhouse_aggregate_functions_cuda PUBLIC
#        "${ClickHouse_SOURCE_DIR}/base/base" "${ClickHouse_BINARY_DIR}/base/base")
#target_link_libraries (clickhouse_aggregate_functions_cuda PUBLIC lib_for_cuda)