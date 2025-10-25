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
    find_package(SndFile CONFIG REQUIRED)
    find_package(OpenBLAS CONFIG REQUIRED)
    find_package(PNG CONFIG REQUIRED)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
    pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)

    # Create imported target for FFTW3 to enable proper dependency propagation
    if(NOT TARGET FFTW3::FFTW3)
        find_library(FFTW3_LIBRARY NAMES fftw3f PATHS ${FFTW3_LIBRARY_DIRS})
        add_library(FFTW3::FFTW3 UNKNOWN IMPORTED)
        set_target_properties(FFTW3::FFTW3 PROPERTIES
            IMPORTED_LOCATION "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
            INTERFACE_LINK_DIRECTORIES "${FFTW3_LIBRARY_DIRS}"
        )
    endif()
    find_package(BLAS REQUIRED)
    find_package(PNG REQUIRED)
    # Find OpenMP
    if(APPLE)
        if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
            set(OPENMP_ROOT "/usr/local/opt/libomp")
        else()
            # For arm64, get the standard brew prefix
            execute_process(COMMAND brew --prefix OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OPENMP_ROOT "${BREW_PREFIX}/opt/libomp")
        endif()
    endif()
    find_package(OpenMP COMPONENTS C REQUIRED)
endif()

# Include directories
include_directories(headers)

# Source files
file(GLOB_RECURSE CARA_SOURCES
    "src/audio_tools/*.c"
    "src/utils/*.c"
    "src/libheatmap/*.c"
    "src/png_tools/*.c"
)
list(FILTER CARA_SOURCES EXCLUDE REGEX "main\\.c$")
list(FILTER CARA_SOURCES EXCLUDE REGEX "test_.*\\.c$")

# Create CARA library
add_library(CARA STATIC ${CARA_SOURCES})

# --- Use target_* commands for modern CMake properties ---
# This is better than global include_directories or link_libraries

target_include_directories(CARA PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/headers"
)

if(MSVC)
    target_compile_options(CARA PUBLIC /openmp)
else()
    find_package(OpenMP COMPONENTS C REQUIRED)
endif()

# System dependencies
if(APPLE)
    target_include_directories(CARA PUBLIC
        ${PNG_INCLUDE_DIRS}
    )

    target_link_directories(CARA PUBLIC
        ${PNG_LIBRARY_DIRS}
    )

target_link_libraries(CARA INTERFACE
    PNG::PNG
)

target_link_libraries(CARA PUBLIC
    FFTW3::FFTW3
    PkgConfig::SNDFILE
    OpenMP::OpenMP_C
)
else()
    target_include_directories(CARA PUBLIC
        ${PNG_INCLUDE_DIRS}
        OpenMP::OpenMP_C
    )

    target_link_directories(CARA PUBLIC
        ${PNG_LIBRARY_DIRS}
        OpenMP::OpenMP_C
    )

if(MSVC)
    target_link_libraries(CARA PUBLIC
    FFTW3::fftw3f
    SndFile::sndfile
    PNG::PNG
    OpenBLAS::OpenBLAS)
else()
        target_link_libraries(CARA PUBLIC
        FFTW3::FFTW3
        PkgConfig::SNDFILE
        PNG::PNG
        OpenMP::OpenMP_C)
    endif()
endif()

# --- Link OpenMP, BLAS/Accelerate, and Math Library ---
if(APPLE)
    target_link_libraries(CARA PUBLIC ${ACCELERATE_FRAMEWORK})
elseif(NOT MSVC)
    target_link_libraries(CARA PUBLIC OpenMP::OpenMP_C ${BLAS_LIBRARIES})
endif()

if(NOT MSVC)
    target_link_libraries(CARA PUBLIC m)
endif()

add_library(CARA::CARA ALIAS CARA)
