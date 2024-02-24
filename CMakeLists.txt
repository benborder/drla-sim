cmake_minimum_required(VERSION 3.14)

# ----------------------------------------------------------------------------
# DRLA Simulator
# ----------------------------------------------------------------------------

project(drla-sim
	VERSION 0.1.0
	DESCRIPTION "DRLA Simulator"
	LANGUAGES CXX
)

# ----------------------------------------------------------------------------
# Add sub directories
# ----------------------------------------------------------------------------

add_subdirectory(sim_agent)
add_subdirectory(sim_train)
add_subdirectory(sim_run)
