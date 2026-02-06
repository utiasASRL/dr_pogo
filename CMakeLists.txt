cmake_minimum_required(VERSION 3.8)
project(dr_pogo)


find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/RadarInfo.msg"
  "msg/LocalMapInfo.msg"
   DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)


ament_package()
