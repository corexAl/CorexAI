cmake_minimum_required(VERSION 3.18)

project(corex_native C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(corex_native SHARED
    src/corex.c
)

target_include_directories(corex_native
    PUBLIC
    include
)
