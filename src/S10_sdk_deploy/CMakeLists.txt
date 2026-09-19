cmake_minimum_required(VERSION 3.10)

project(s10_sdk_deploy)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 默认构建类型
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# 警告忽略和编译选项
add_definitions(-w)
add_compile_options(-fPIC)

# 配置选项
set(BUILD_PLATFORM "x86" CACHE STRING "Target platform (x86 or arm)")
set(S10_COMMAND_INTERFACE "keyboard" CACHE STRING "Command interface: keyboard, gamepad, or dds")
set_property(CACHE S10_COMMAND_INTERFACE PROPERTY STRINGS keyboard gamepad dds)

if(S10_COMMAND_INTERFACE STREQUAL "dds")
  add_compile_definitions(S10_COMMAND_INTERFACE_DDS)
elseif(S10_COMMAND_INTERFACE STREQUAL "gamepad")
  add_compile_definitions(S10_COMMAND_INTERFACE_GAMEPAD)
elseif(NOT S10_COMMAND_INTERFACE STREQUAL "keyboard")
  message(FATAL_ERROR "S10_COMMAND_INTERFACE must be keyboard, gamepad, or dds")
endif()

if(BUILD_PLATFORM STREQUAL "arm")
  set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
  set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++") 
endif()

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(drdds REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

include_directories(
  include/types
  include/utils
  interface/robot
  interface/user_command
  interface/network
  state_machine
  run_policy
  ${THIRD_PARTY}/eigen/
  ${THIRD_PARTY}/gamepad/include
  ${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}/include
)

# 子目录
add_subdirectory(interface)

# 运行时路径设置
set(CMAKE_EXE_LINKER_FLAGS "-Wl,--disable-new-dtags")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


set(CMAKE_INSTALL_RPATH 
  ${THIRD_PARTY}/onnxruntime/${BUILD_PLATFORM}/lib/
  ../lib/onnxruntime/
)

file(GLOB_RECURSE STATE_MACHINE_SRC "state_machine/*.c*")

add_executable(rl_deploy main.cpp ${STATE_MACHINE_SRC})
add_executable(sru_high_level_dry_run run_policy/sru_high_level_dry_run.cpp)
add_executable(s10_low_level_dry_run run_policy/s10_low_level_dry_run.cpp)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${THIRD_PARTY}/onnxruntime//${BUILD_PLATFORM}/lib/libonnxruntime.so")

ament_target_dependencies(rl_deploy
  rclcpp
  drdds
  geometry_msgs
  std_msgs
)
ament_target_dependencies(sru_high_level_dry_run
  rclcpp
  drdds
  geometry_msgs
  std_msgs
)
target_link_libraries(sru_high_level_dry_run onnxruntime)
ament_target_dependencies(s10_low_level_dry_run
  rclcpp
  drdds
)
target_link_libraries(s10_low_level_dry_run onnxruntime)
if(BUILD_PLATFORM STREQUAL "arm")
  target_link_libraries(rl_deploy
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
else()
  target_link_libraries(rl_deploy
    interface
    onnxruntime
    -lpthread
    -lm
    -lrt
    -ldl
    -lstdc++
  )
endif()


install(TARGETS rl_deploy sru_high_level_dry_run s10_low_level_dry_run
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  interface/navigation/waypoint_follower.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME waypoint_follower
)

ament_package()
