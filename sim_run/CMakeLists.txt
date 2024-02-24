cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

include(${CMAKE_SOURCE_DIR}/cmake/cxxopts.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/spdlog.cmake)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

# ----------------------------------------------------------------------------
# Building Simulator run cli
# ----------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

add_executable(sim_run
	src/main.cpp
	src/runner.cpp
)

# Using PRIVATE in target_compile_options keeps the options local to this library
target_compile_options(sim_run PRIVATE -Wall -Wextra -Werror -Wno-unused $<$<CONFIG:RELEASE>:-O2 -flto>)
target_compile_features(sim_run PRIVATE cxx_std_17)

target_include_directories(sim_run
	PRIVATE
		src
)

target_link_libraries(sim_run
PUBLIC
	sim_agent
	${TORCH_LIBRARIES}
	Threads::Threads
	cxxopts
	spdlog
)

# ----------------------------------------------------------------------------
# Installing Simulator train cli
# ----------------------------------------------------------------------------

install(
	TARGETS sim_run
	RUNTIME DESTINATION bin
)
