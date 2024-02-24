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
# Building Simulator train cli
# ----------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

add_executable(sim_train
	src/main.cpp
	src/logger.cpp
)

# Using PRIVATE in target_compile_options keeps the options local to this library
target_compile_options(sim_train PRIVATE -Wall -Wextra -Werror -Wno-unused $<$<CONFIG:RELEASE>:-O2 -flto>)
target_compile_features(sim_train PRIVATE cxx_std_17)

target_include_directories(sim_train
	PRIVATE
		src
)

target_link_libraries(sim_train
PUBLIC
	sim_agent
	${TORCH_LIBRARIES}
	Threads::Threads
	cxxopts
	spdlog
)

# ----------------------------------------------------------------------------
# Installing simulator train cli
# ----------------------------------------------------------------------------

install(
	TARGETS sim_train
	RUNTIME DESTINATION bin
)
