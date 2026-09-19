cmake_minimum_required(VERSION 3.16)
project(s10_multilap_pose_graph LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")

find_package(nlohmann_json REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gtsam_points REQUIRED)
if(TARGET metis-gtsam-if AND
   NOT EXISTS "/usr/lib/x86_64-linux-gnu/libmetis.so" AND
   EXISTS "/usr/lib/x86_64-linux-gnu/libmetis.so.5")
  # The locally installed GTSAM export references the development-package
  # symlink, while the runtime METIS library is present. Keep the workaround
  # project-local instead of modifying the host system.
  set_target_properties(metis-gtsam-if PROPERTIES
    INTERFACE_LINK_LIBRARIES "/usr/lib/x86_64-linux-gnu/libmetis.so.5")
endif()
find_path(GTSAM_INCLUDE_DIR gtsam/geometry/Pose3.h
  PATHS /usr/local/include REQUIRED)
find_library(GTSAM_LIBRARY gtsam
  PATHS /usr/local/lib NO_DEFAULT_PATH REQUIRED)

add_executable(multilap_pose_graph multilap_pose_graph.cpp)
target_include_directories(multilap_pose_graph PRIVATE ${GTSAM_INCLUDE_DIR})
target_link_libraries(multilap_pose_graph PRIVATE
  ${GTSAM_LIBRARY}
  Eigen3::Eigen
  nlohmann_json::nlohmann_json)
set_target_properties(multilap_pose_graph PROPERTIES
  BUILD_RPATH "/usr/local/lib")

find_package(Python3 3.10 EXACT COMPONENTS Interpreter Development.Module QUIET)
find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h
  HINTS "${Python3_SITEARCH}/torch/include")
if(Python3_FOUND AND PYBIND11_INCLUDE_DIR)
  add_library(s10_vgicp MODULE vgicp_python.cpp)
  target_include_directories(s10_vgicp PRIVATE ${PYBIND11_INCLUDE_DIR})
  target_link_libraries(s10_vgicp PRIVATE
    gtsam_points::gtsam_points
    Python3::Module)
  set_target_properties(s10_vgicp PROPERTIES
    PREFIX ""
    BUILD_RPATH "/usr/local/lib")
endif()
