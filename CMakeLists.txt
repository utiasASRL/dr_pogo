cmake_minimum_required(VERSION 3.8)
project(dr_pogo)


find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

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


ament_package()
