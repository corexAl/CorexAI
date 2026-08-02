cmake_minimum_required(VERSION 3.15)

project(corex_native)

find_package(pybind11 REQUIRED)

pybind11_add_module(
    corex_native
    bindings.cpp
    corex.cpp
)
