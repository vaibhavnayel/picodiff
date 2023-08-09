cmake_minimum_required(VERSION 3.10)
project( main )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package (Eigen3 3.4 REQUIRED NO_MODULE)
include_directories(${Eigen_INCLUDE_DIRS})

add_executable( main main.cpp src/autograd.cpp src/ops.cpp)
target_link_libraries( main Eigen3::Eigen )
target_include_directories(main PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/src")