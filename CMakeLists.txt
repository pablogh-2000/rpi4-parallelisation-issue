#*************************************************************************
#
#                           Klepsydra Core Modules
#              Copyright (C) 2019-2020  Klepsydra Technologies GmbH
#                            All Rights Reserved.
#
#  This file is subject to the terms and conditions defined in
#  file 'LICENSE.md', which is part of this source code package.
#
#  NOTICE:  All information contained herein is, and remains the property of Klepsydra
#  Technologies GmbH and its suppliers, if any. The intellectual and technical concepts
#  contained herein are proprietary to Klepsydra Technologies GmbH and its suppliers and
#  may be covered by Swiss and Foreign Patents, patents in process, and are protected by
#  trade secret or copyright law. Dissemination of this information or reproduction of
#  this material is strictly forbidden unless prior written permission is obtained from
#  Klepsydra Technologies GmbH.
#
#*************************************************************************
cmake_minimum_required(VERSION 3.16)

set(PROJ_NAME kpsr_matrix_mult_benchmark)
project(${PROJ_NAME})

# Source
# ---------------------------------------------------#
file(GLOB ${PROJ_NAME}_SRC "src/*.cpp")

file(GLOB_RECURSE ${PROJ_NAME}_HEADERS "include/*.h")

# Create Library
# ---------------------------------------------------#
add_executable(${PROJ_NAME} src/multiplier_test.cpp ${${PROJ_NAME}_HEADERS} )

set(MATRIX_COMPILE_OPTIONS
    "-march=native;-funsafe-math-optimizations;-ftree-vectorize;-fomit-frame-pointer"
)
target_compile_options(
    ${PROJ_NAME} INTERFACE $<$<COMPILE_LANGUAGE:CXX>:${MATRIX_COMPILE_OPTIONS}>)

# Link libraries with Project
# ---------------------------------------------------#

target_include_directories(
  ${PROJ_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

find_package(Eigen3 QUIET)
find_package(OpenMP QUIET)
find_package(BLAS QUIET)
find_package(Threads REQUIRED)

if(KPSR_WITH_OPENMP)
  list(APPEND MATH_LIBRARIES OpenMP::OpenMP_CXX)
  list(APPEND KPSR_COMPILE_DEFINITIONS openmp_enabled)
  target_compile_options(${PROJ_NAME} INTERFACE -fopenmp)
elseif(KPSR_WITH_OPENBLAS)
  list(APPEND KPSR_COMPILE_DEFINITIONS blas_enabled)
  list(APPEND MATH_LIBRARIES ${BLAS_LIBRARIES})
elseif(KPSR_WITH_EIGEN)
  list(APPEND KPSR_COMPILE_DEFINITIONS eigen_enabled
    EIGEN_DONT_PARALLELIZE)
  list(APPEND MATH_LIBRARIES Eigen3::Eigen)
elseif(KPSR_WITH_EIGEN_PARALLEL)
  list(APPEND KPSR_COMPILE_DEFINITIONS eigen_enabled openmp_enabled)
  list(APPEND MATH_LIBRARIES Eigen3::Eigen OpenMP::OpenMP_CXX)
  target_compile_options(${PROJ_NAME}  INTERFACE -fopenmp)
elseif(KPSR_WITH_RUY)
  list(APPEND KPSR_COMPILE_DEFINITIONS ruy_enabled)
  list(APPEND MATH_LIBRARIES ruy)
endif()

target_compile_definitions(${PROJ_NAME}
  PUBLIC ${KPSR_COMPILE_DEFINITIONS})
message("MATH_LIBRARIES: ${MATH_LIBRARIES}")
target_link_libraries(${PROJ_NAME} PUBLIC ${CMAKE_THREAD_LIBS_INIT})
# PRINTBASICINFO(${PROJ_NAME})
