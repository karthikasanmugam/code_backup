cmake_minimum_required(VERSION 3.10)
project(daksha_vla_ui)

find_package(ament_cmake REQUIRED)

install(
  PROGRAMS scripts/daksha_vla_ui
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
