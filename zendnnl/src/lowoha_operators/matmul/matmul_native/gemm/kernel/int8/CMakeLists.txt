# INT8 GEMM microkernels
set(sources
    int8_symq_ukernel_128.cpp
    int8_kquant_ukernel_128.cpp
    int8_q4k_gemv_128.cpp
    int8_q4k_gemv_ilv_128.cpp
    int8_q6k_gemv_pre_128.cpp
    int8_q4k_gemv_ilv8_256.cpp)

set(headers
    int8_symq_ukernel_128.hpp
    int8_kquant_ukernel_128.hpp
    int8_q4k_gemv_128.hpp
    int8_q4k_gemv_ilv_128.hpp
    int8_q6k_gemv_pre_128.hpp
    int8_q4k_gemv_ilv8_256.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
    PRIVATE ${sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
    PRIVATE ${sources}
    PRIVATE FILE_SET zendnnl_pvt_headers
            TYPE HEADERS FILES ${headers})
endif()
