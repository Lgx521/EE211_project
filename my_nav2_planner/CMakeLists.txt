cmake_minimum_required(VERSION 3.8)
project(my_nav2_planner)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_lifecycle REQUIRED)
find_package(nav2_core REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

include_directories(include)

# 创建库
add_library(my_astar_planner_plugin SHARED
  src/my_astar_planner.cpp
)

# 链接依赖
ament_target_dependencies(my_astar_planner_plugin
  rclcpp
  rclcpp_lifecycle
  nav2_core
  nav2_costmap_2d
  nav2_util
  pluginlib
  geometry_msgs
  nav_msgs
)

# 注册插件 XML
pluginlib_export_plugin_description_file(nav2_core global_planner_plugin.xml)

# 安装头文件和库
install(TARGETS my_astar_planner_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include/
)

install(FILES global_planner_plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

ament_export_libraries(my_astar_planner_plugin)
ament_export_include_directories(include)
ament_package()