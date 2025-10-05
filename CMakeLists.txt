cmake_minimum_required(VERSION 3.27)

project(CARA C)
set(CMAKE_C_STANDARD 11)

# Compiler flags for optimization (matching CARA's Makefile)
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops -fpeel-loops -ftracer -flto -fuse-linker-plugin -floop-block -floop-interchange -floop-unroll-and-jam -fipa-pta -fipa-cp -fipa-sra -fipa-icf -fno-unsafe-math-optimizations")

# Vectorization flags
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ftree-vectorize -ftree-loop-vectorize")

# Library flags
add_definitions(-DMINIMP3_FLOAT_OUTPUT -DLOG_LEVEL=1 -DBUILTIN)

# Find required packages
find_package(PkgConfig REQUIRED)

# Find FFTW3
pkg_check_modules(FFTW3 REQUIRED fftw3 fftw3f)

# Find libsndfile
pkg_check_modules(SNDFILE REQUIRED sndfile)

# Find OpenBLAS
find_package(BLAS REQUIRED)

# Add OpenBLAS include directory
if(EXISTS "/usr/include/openblas")
    include_directories("/usr/include/openblas")
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find PNG (for visualization features)
find_package(PNG REQUIRED)

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

# Link libraries
target_link_libraries(CARA 
    ${FFTW3_LIBRARIES}
    ${SNDFILE_LIBRARIES}
    ${BLAS_LIBRARIES}
    PNG::PNG
    OpenMP::OpenMP_C
    m
)

# Include directories for linking
target_include_directories(CARA PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/headers
    ${FFTW3_INCLUDE_DIRS}
    ${SNDFILE_INCLUDE_DIRS}
)

# Add OpenBLAS include directory if it exists
if(EXISTS "/usr/include/openblas")
    target_include_directories(CARA PRIVATE "/usr/include/openblas")
endif()

# Compiler flags
target_compile_options(CARA PRIVATE 
    ${FFTW3_CFLAGS_OTHER}
    ${SNDFILE_CFLAGS_OTHER}
)

# Create alias for easier linking
add_library(CARA::CARA ALIAS CARA)
