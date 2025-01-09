cmake_minimum_required(VERSION 3.22)

set( CMAKE_C_COMPILER "gcc" )
set( CMAKE_CXX_COMPILER "g++" )

project(HelloLBM)

find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(FreeGLUT CONFIG REQUIRED)

add_executable(hello-lbm main.cpp ShaderProgram.cpp)

target_link_libraries(hello-lbm PRIVATE glfw glad::glad fmt::fmt glm::glm $<IF:$<TARGET_EXISTS:FreeGLUT::freeglut>,FreeGLUT::freeglut,FreeGLUT::freeglut_static> GLU)
