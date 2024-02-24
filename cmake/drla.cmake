find_package(drla QUIET)
if(${drla_FOUND})
	message(STATUS "Found drla ${drla_DIR}")
else()
	include(FetchContent)
	FetchContent_Declare(
		drla
		GIT_REPOSITORY https://github.com/benborder/drla.git
		GIT_TAG        master
	)
	FetchContent_Populate(drla)
	add_subdirectory(${drla_SOURCE_DIR} ${drla_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
