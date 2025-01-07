cmake_minimum_required(VERSION 3.22)

set( CMAKE_C_COMPILER "gcc" )
set( CMAKE_CXX_COMPILER "g++" )

project(HelloShaderToy)

find_package(glad CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

add_executable(hello-shadertoy main.cpp ShaderProgram.cpp)

target_link_libraries(hello-shadertoy PRIVATE glfw glad::glad fmt::fmt)
