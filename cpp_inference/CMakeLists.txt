cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(TorchInference LANGUAGES CXX)

# Use C++17 (required by LibTorch)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to LibTorch
set(Torch_DIR "/home/ubuntu/projects/torch/libtorch/share/cmake/Torch")

find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)

add_executable(inference inference.cpp)

# Link against LibTorch
target_link_libraries(inference "${TORCH_LIBRARIES}")
target_link_libraries(inference ${OpenCV_LIBS})

set_property(TARGET inference PROPERTY CXX_STANDARD 17)

# Optional: Make sure rpath is set so it can find the .so libs
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
