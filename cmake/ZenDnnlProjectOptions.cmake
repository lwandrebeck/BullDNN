# *******************************************************************************
# * Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
include_guard(GLOBAL)

set(ZENDNNL_DEPS_DIR       "${ZENDNNL_SOURCE_DIR}/dependencies"
  CACHE PATH "zendnnl dependencies dir")
set(ZENDNNL_BINARY_DIR     "${ZENDNNL_SOURCE_DIR}/build"
  CACHE PATH "zendnnl build dir")
set(ZENDNNL_INSTALL_PREFIX "${ZENDNNL_BINARY_DIR}/install"
  CACHE PATH "zendnnl install prefix")

set(ZENDNNL_VERBOSE_MAKEFILE ON CACHE BOOL "zendnnl verbose makefile")
set(ZENDNNL_MESSAGE_LOG_LEVEL "DEBUG" CACHE STRING "zendnnl log level")
set(ZENDNNL_BUILD_TYPE "Release" CACHE STRING "zendnnl build type")
set(ZENDNNL_MSG_PREFIX "(ZENDNNL) " CACHE STRING "zendnnl message prefix")

# Target CPU architecture, forwarded to the zendnnl library sub-build
# (consumed in zendnnl/CMakeLists.txt). Default "zen" preserves upstream
# behavior; bdver1..bdver4 target AMD family 15h; native tunes for the host.
set(BULLDNN_TARGET_ARCH "zen" CACHE STRING
  "Target CPU arch: zen | bdver1 | bdver2 | bdver3 | bdver4 | native")
set_property(CACHE BULLDNN_TARGET_ARCH PROPERTY STRINGS
  zen bdver1 bdver2 bdver3 bdver4 native)
message(STATUS "${ZENDNNL_MSG_PREFIX}BULLDNN_TARGET_ARCH=${BULLDNN_TARGET_ARCH}")

# cmake variables
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} ${CMAKE_MODULE_PATH}")
set(CMAKE_INSTALL_PREFIX "${ZENDNNL_INSTALL_PREFIX}")
set(CMAKE_VERBOSE_MAKEFILE "${ZENDNNL_VERBOSE_MAKEFILE}")
set(CMAKE_MESSAGE_LOG_LEVEL "${ZENDNNL_MESSAGE_LOG_LEVEL}")
set(CMAKE_BUILD_TYPE "${ZENDNNL_BUILD_TYPE}")

# informative messages
message(STATUS "${ZENDNNL_MSG_PREFIX}ZENDNNL_SOURCE_DIR=${ZENDNNL_SOURCE_DIR}")
message(STATUS "${ZENDNNL_MSG_PREFIX}ZENDNNL_BINARY_DIR=${ZENDNNL_BINARY_DIR}")
message(STATUS "${ZENDNNL_MSG_PREFIX}ZENDNNL_DEPS_DIR=${ZENDNNL_DEPS_DIR}")

message(STATUS "${ZENDNNL_MSG_PREFIX}CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

message(STATUS "${ZENDNNL_MSG_PREFIX}CMAKE_VERBOSE_MAKEFILE=${CMAKE_VERBOSE_MAKEFILE}")
message(STATUS "${ZENDNNL_MSG_PREFIX}CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
message(STATUS "${ZENDNNL_MSG_PREFIX}CMAKE_MESSAGE_LOG_LEVEL=${CMAKE_MESSAGE_LOG_LEVEL}")

