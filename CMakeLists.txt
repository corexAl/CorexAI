cmake_minimum_required(VERSION 3.18)

project(COREX
    VERSION 0.1.0
    DESCRIPTION "COREX AI Native Engine"
    LANGUAGES CXX
)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build options
option(COREX_BUILD_NATIVE "Build COREX native acceleration module" ON)

if(COREX_BUILD_NATIVE)

    message(STATUS "Building COREX native module")

    # Find Python
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Find pybind11
    find_package(pybind11 REQUIRED)

    # Native module
    pybind11_add_module(
        corex_native

        native/bindings.cpp
        native/corex.cpp
    )

    target_include_directories(
        corex_native
        PRIVATE
        native
    )

    # Optimization flags
    if(MSVC)
        target_compile_options(corex_native PRIVATE /O2)
    else()
        target_compile_options(corex_native PRIVATE -O3)
    endif()

endif()

# Install Python package files
install(
    DIRECTORY src/
    DESTINATION .
)
