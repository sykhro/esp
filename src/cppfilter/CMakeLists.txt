# Elia Melucci - (c) 2020
# Esperienze di Programmazione, UniPi

cmake_minimum_required(VERSION 3.5)

project(cppfilter LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

add_executable(sigtaint sigtaint.cpp)
target_compile_features(sigtaint INTERFACE cxx_std_17)

find_package(FFTW REQUIRED)

add_executable(sigclean
    sigclean.cpp
    gnuplot_i.c
)
target_link_libraries(sigclean PRIVATE ${FFTW_LIBRARIES})
target_include_directories(sigclean PRIVATE ${FFTW_INCLUDES} .)
target_compile_features(sigclean INTERFACE cxx_std_17)
