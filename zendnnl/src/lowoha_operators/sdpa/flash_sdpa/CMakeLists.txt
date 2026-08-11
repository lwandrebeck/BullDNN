# *******************************************************************************
# * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# *******************************************************************************/

# ---------------------------------------------------------------------------
#  No per-file ISA flags needed: lowoha_operators/common/simd_ops.hpp provides
#  AVX-512 target attributes/wrappers (for example,
#  __attribute__((target("avx512f,avx512bw,avx512vl,fma")))), while
#  runtime ISA dispatch is handled in the flash SDPA implementation (for
#  example, lowoha_sdpa_flash_cpu.cpp).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Sources and headers
# ---------------------------------------------------------------------------
# lowoha_sdpa_flash_cpu_scalar.cpp rebuilds lowoha_sdpa_flash_cpu.cpp with
# ZENDNNL_SDPA_FLASH_TIER_SCALAR set, i.e. without the AVX-512 target pragma,
# so the scalar fallback kernels carry no FMA3 (which AMD Bulldozer lacks).
# See the comment at the top of that file.
set(lowoha_flash_sdpa_sources
    lowoha_flash_sdpa.cpp
    lowoha_flash_sdpa_utils.cpp
    lowoha_sdpa_flash_cpu.cpp
    lowoha_sdpa_flash_cpu_scalar.cpp)

set(lowoha_flash_sdpa_public_headers
    lowoha_flash_sdpa.hpp
    lowoha_flash_sdpa_utils.hpp
    lowoha_sdpa_flash_cpu.hpp)

if(ZENDNNL_LIB_BUILD_ARCHIVE)
  target_sources(zendnnl_archive
                 PRIVATE ${lowoha_flash_sdpa_sources}
                 PUBLIC  FILE_SET zendnnl_headers
                 TYPE HEADERS FILES ${lowoha_flash_sdpa_public_headers})
endif()

if(ZENDNNL_LIB_BUILD_SHARED)
  target_sources(zendnnl
                 PRIVATE ${lowoha_flash_sdpa_sources}
                 PUBLIC  FILE_SET zendnnl_headers
                 TYPE HEADERS FILES ${lowoha_flash_sdpa_public_headers})
endif()

# append header files to top level include file
set(inc_subdir ${CMAKE_CURRENT_SOURCE_DIR})
cmake_path(RELATIVE_PATH inc_subdir
           BASE_DIRECTORY ${CMAKE_SOURCE_DIR}/src)

list(APPEND lowoha_flash_sdpa_public_headers_list
     ${lowoha_flash_sdpa_public_headers})
create_header_file(${zendnnl_inc_file}
                   ${inc_subdir}
                   "${lowoha_flash_sdpa_public_headers_list}")
