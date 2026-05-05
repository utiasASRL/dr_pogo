cmake_minimum_required(VERSION 3.8)
project(dr_pogo)


find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(std_msgs REQUIRED)

install(PROGRAMS
  dr_pogo/boreas_player.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME boreas_player
)

install(PROGRAMS
  dr_pogo/dro_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME dro_node
)

install(PROGRAMS
  dr_pogo/raplace_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME raplace_node
)

install(PROGRAMS
  dr_pogo/registration_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME registration_node
)

install(FILES
  dr_pogo/dro.py
  dr_pogo/dro_motion_models.py
  DESTINATION lib/${PROJECT_NAME}
)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadarInfo.msg"
  "msg/LocalMapInfo.msg"
  "msg/LoopCandidate.msg"
   DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

add_executable(pogo_node
  dr_pogo/pogo/pogo_node.cpp dr_pogo/pogo/pose_graph.cpp
)
target_include_directories(pogo_node PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/dr_pogo/pogo
  ${YAML_CPP_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(pogo_node
  rclcpp
  nav_msgs
  geometry_msgs
)

# Non-ament CMake packages must be linked explicitly.
if(TARGET yaml-cpp::yaml-cpp)
  target_link_libraries(pogo_node yaml-cpp::yaml-cpp)
else()
  target_link_libraries(pogo_node yaml-cpp)
endif()

if(TARGET Ceres::ceres)
  target_link_libraries(pogo_node Ceres::ceres)
else()
  target_link_libraries(pogo_node ${CERES_LIBRARIES})
endif()

install(TARGETS pogo_node
  DESTINATION lib/${PROJECT_NAME}
)


ament_package()
