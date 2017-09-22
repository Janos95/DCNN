
cmake_minimum_required(VERSION 2.8)
project(deformableconvolution)

find_package(Torch)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")


if(${Torch_FOUND})
	message(STATUS "Building torch module")
	add_torch_package(deformableconvolution
	ccode/deformableconvolution.cpp
		"init.lua"
		"DeformableConvolution"
	)
	target_link_libraries(deformableconvolution
		luaT TH
	)
endif()
