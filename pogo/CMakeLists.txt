cmake_minimum_required(VERSION 3.10)
project(pogo)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

# Create pose_graph library from src
add_library(pose_graph
    src/pose_graph.cpp
)

# Link pose_graph library with Ceres
target_link_libraries(pose_graph
    ${CERES_LIBRARIES}
)

# Set include directories for the library
target_include_directories(pose_graph PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CERES_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
)

# Create pogo executable from app
add_executable(pogo
    app/pogo.cpp
)

# Link pogo executable with pose_graph library and Ceres
target_link_libraries(pogo
    pose_graph
    ${CERES_LIBRARIES}
    yaml-cpp
)

# Set compile flags for better optimization and debugging
target_compile_options(pose_graph PRIVATE -Wall -Wextra -O3)
target_compile_options(pogo PRIVATE -Wall -Wextra -O3)


