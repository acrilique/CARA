cmake_minimum_required(VERSION 3.27)

project(CARA C)
set(CMAKE_C_STANDARD 11)

# Compiler flags for optimization (matching CARA's Makefile)
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops -fpeel-loops -ftracer -flto -fuse-linker-plugin -floop-block -floop-interchange -floop-unroll-and-jam -fipa-pta -fipa-cp -fipa-sra -fipa-icf -fno-unsafe-math-optimizations")
    # Vectorization flags
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ftree-vectorize -ftree-loop-vectorize")
endif()

# Library flags
add_definitions(-DMINIMP3_FLOAT_OUTPUT -DLOG_LEVEL=1 -DBUILTIN)

if(CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES ".*/vcpkg.cmake")
    find_package(FFTW3 CONFIG REQUIRED)
    find_package(sndfile CONFIG REQUIRED libsndfile)
    find_package(OpenBLAS CONFIG REQUIRED)
    find_package(PNG CONFIG REQUIRED)
else()
    # Find required packages
    find_package(PkgConfig REQUIRED)

    # Find FFTW3
    pkg_check_modules(FFTW3 REQUIRED fftw3 fftw3f)

    # Find libsndfile
    pkg_check_modules(SNDFILE REQUIRED sndfile)

    # Find OpenBLAS
    find_package(BLAS REQUIRED)

    # Find PNG (for visualization features)
    find_package(PNG REQUIRED)
endif()

# omp on macos
if(EXISTS "/opt/homebrew/opt/libomp/include")
    message(STATUS "Found libomp include directory")
    include_directories("/opt/homebrew/opt/libomp/include")
endif()

# Find OpenMP
if(NOT MSVC)
    find_package(OpenMP REQUIRED)
endif()

# Include directories
include_directories(headers)

# Source files - include all CARA sources
file(GLOB_RECURSE CARA_SOURCES
    "src/audio_tools/*.c"
    "src/utils/*.c"
    "src/libheatmap/*.c"
    "src/png_tools/*.c"
)

# Exclude main.c and test files
list(FILTER CARA_SOURCES EXCLUDE REGEX "main\\.c$")
list(FILTER CARA_SOURCES EXCLUDE REGEX "test_.*\\.c$")

# Create CARA library
add_library(CARA STATIC ${CARA_SOURCES})

if(EXISTS "/usr/include/openblas")
    target_include_directories(CARA PUBLIC "/usr/include/openblas")
endif()
if(EXISTS "/opt/homebrew/opt/openblas/include")
    target_include_directories(CARA PUBLIC "/opt/homebrew/opt/openblas/include")
endif()

if(MSVC)
    target_compile_options(CARA PUBLIC /openmp)
endif()

# Link libraries
if(CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES ".*/vcpkg.cmake")
    target_link_libraries(CARA PUBLIC
        FFTW3::fftw3
        sndfile::sndfile
        OpenBLAS::OpenBLAS
        PNG::PNG
    )
else()
    target_link_libraries(CARA PUBLIC
        ${FFTW3_LIBRARIES}
        ${SNDFILE_LIBRARIES}
        ${BLAS_LIBRARIES}
        PNG::PNG
    )
endif()

if(NOT MSVC)
    target_link_libraries(CARA PUBLIC OpenMP::OpenMP_C m)
endif()

# Include directories for linking
if(CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES ".*/vcpkg.cmake")
    target_include_directories(CARA PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/headers
    )
else()
    target_include_directories(CARA PUBLIC 
        ${CMAKE_CURRENT_SOURCE_DIR}/headers
        ${FFTW3_INCLUDE_DIRS}
        ${SNDFILE_INCLUDE_DIRS}
    )
endif()


if(NOT (CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES ".*/vcpkg.cmake"))
    # Compiler flags
    target_compile_options(CARA PUBLIC 
        ${FFTW3_CFLAGS_OTHER}
        ${SNDFILE_CFLAGS_OTHER}
    )
endif()

# Create alias for easier linking
add_library(CARA::CARA ALIAS CARA)
