cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# Simulator agent
# ----------------------------------------------------------------------------

project(sim_agent
	VERSION 0.1.0
	DESCRIPTION "Simulator agent"
	LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Dependencies
# ----------------------------------------------------------------------------

include(${CMAKE_SOURCE_DIR}/cmake/drla.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/spdlog.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/nlohmann_json.cmake)

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# ----------------------------------------------------------------------------
# Building Simulator
# ----------------------------------------------------------------------------

set(THREADS_PREFER_PTHREAD_FLAG ON)  # Set -pthreads
find_package(Threads REQUIRED)
find_package(Torch REQUIRED)

add_library(sim_agent STATIC
  src/sim_agent.cpp
  src/classic_control/cartpole_env.cpp
  src/abstract_strategy/tictactoe_env.cpp
  src/abstract_strategy/connectfour_env.cpp
  src/abstract_strategy/gridworld_env.cpp
  src/mcts/mcts.cpp
  src/render/render.cpp
  src/utility.cpp
)

# Using PRIVATE in target_compile_options keeps the options local to this library
target_compile_options(sim_agent PRIVATE -Wall -Wextra -Werror -Wno-unused -Wpedantic $<$<CONFIG:RELEASE>:-O2 -flto>)
target_compile_features(sim_agent PRIVATE cxx_std_17)

target_include_directories(sim_agent
  PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/sim_agent>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/classic_control>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/abstract_strategy>
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
)

target_link_libraries(sim_agent
PUBLIC
  drla::drla
  drla::aux
  ${TORCH_LIBRARIES}
  Threads::Threads
  nlohmann_json::nlohmann_json
PRIVATE
  $<BUILD_INTERFACE:spdlog::spdlog>
  $<BUILD_INTERFACE:${OpenCV_LIBS}>
)

# ----------------------------------------------------------------------------
# Installing the Simulator agent library
# ----------------------------------------------------------------------------

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

set(sim_agent_export_targets sim_agent)
if(NOT nlohmann_json_FOUND)
	set(sim_agent_export_targets ${sim_agent_export_targets} nlohmann_json)
endif()

install(
  TARGETS ${sim_agent_export_targets}
  EXPORT sim_agent_Targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  INCLUDES DESTINATION include
)

write_basic_package_version_file(
  "sim_agentConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/sim_agentConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/sim_agentConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
  EXPORT sim_agent_Targets
  FILE sim_agentTargets.cmake
  NAMESPACE sim_agent::
  DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
  FILES "${PROJECT_BINARY_DIR}/sim_agentConfig.cmake"
        "${PROJECT_BINARY_DIR}/sim_agentConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_PREFIX}/cmake
)

install(
  DIRECTORY include/
  DESTINATION include
)
