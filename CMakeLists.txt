cmake_minimum_required(VERSION 3.27)

project(CARA C)
set(CMAKE_C_STANDARD 11)

# --- Compiler Flags ---
if(CMAKE_C_COMPILER_ID MATCHES "GNU")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -march=native -mtune=native -funroll-loops -fpeel-loops -ftracer -flto -fuse-linker-plugin -floop-block -floop-interchange -floop-unroll-and-jam -fipa-pta -fipa-cp -fipa-sra -fipa-icf -fno-unsafe-math-optimizations")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -ftree-vectorize -ftree-loop-vectorize")
endif()

add_definitions(-DMINIMP3_FLOAT_OUTPUT -DLOG_LEVEL=1 -DBUILTIN)

# --- Source Files ---
file(GLOB_RECURSE CARA_SOURCES
    "src/audio_tools/*.c"
    "src/utils/*.c"
    "src/libheatmap/*.c"
    "src/png_tools/*.c"
)
list(FILTER CARA_SOURCES EXCLUDE REGEX "main\\.c$")
list(FILTER CARA_SOURCES EXCLUDE REGEX "test_.*\\.c$")

# --- Library Definition ---
add_library(CARA STATIC ${CARA_SOURCES})
add_library(CARA::CARA ALIAS CARA)

target_include_directories(CARA PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/headers")

if(MSVC)
    target_compile_options(CARA PUBLIC /openmp)
endif()

# --- Dependencies ---
set(CARA_INCLUDE_DIRS "")
set(CARA_LINK_LIBRARIES "")

if(CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES ".*/vcpkg.cmake")
    find_package(FFTW3 CONFIG REQUIRED)
    find_package(SndFile CONFIG REQUIRED)
    find_package(OpenBLAS CONFIG REQUIRED)
    find_package(PNG CONFIG REQUIRED)

    list(APPEND CARA_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS})
    list(APPEND CARA_LINK_LIBRARIES ${FFTW3_LIBRARIES} SndFile::sndfile OpenBLAS::OpenBLAS PNG::PNG)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFTW3 REQUIRED fftw3f)
    pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
    find_package(BLAS REQUIRED)
    find_package(PNG REQUIRED)

    if(NOT TARGET FFTW3::FFTW3)
        find_library(FFTW3_LIBRARY NAMES fftw3f PATHS ${FFTW3_LIBRARY_DIRS})
        add_library(FFTW3::FFTW3 UNKNOWN IMPORTED)
        set_target_properties(FFTW3::FFTW3 PROPERTIES
            IMPORTED_LOCATION "${FFTW3_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIRS}"
        )
    endif()

    list(APPEND CARA_INCLUDE_DIRS ${FFTW3_INCLUDE_DIRS} ${SNDFILE_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
    list(APPEND CARA_LINK_LIBRARIES FFTW3::FFTW3 PkgConfig::SNDFILE PNG::PNG ${BLAS_LIBRARIES})
endif()

# --- Platform-specific Dependencies ---
if(APPLE)
    if(NOT (CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES ".*/vcpkg.cmake"))
        if(CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
            set(OPENMP_ROOT "/usr/local/opt/libomp")
        else()
            execute_process(COMMAND brew --prefix OUTPUT_VARIABLE BREW_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
            set(OPENMP_ROOT "${BREW_PREFIX}/opt/libomp")
        endif()
    endif()
    find_package(OpenMP)
    if(OpenMP_FOUND)
        list(APPEND CARA_INCLUDE_DIRS ${OpenMP_C_INCLUDE_DIRS})
        list(APPEND CARA_LINK_LIBRARIES OpenMP::OpenMP_C)
    endif()
    find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)
    list(APPEND CARA_LINK_LIBRARIES ${ACCELERATE_FRAMEWORK})
elseif(NOT MSVC)
    find_package(OpenMP COMPONENTS C REQUIRED)
    list(APPEND CARA_INCLUDE_DIRS ${OpenMP_C_INCLUDE_DIRS})
    list(APPEND CARA_LINK_LIBRARIES OpenMP::OpenMP_C)
    target_link_libraries(CARA PUBLIC m)
endif()

# --- Apply Dependencies to Target ---
target_include_directories(CARA PUBLIC ${CARA_INCLUDE_DIRS})
target_link_libraries(CARA PUBLIC ${CARA_LINK_LIBRARIES})
