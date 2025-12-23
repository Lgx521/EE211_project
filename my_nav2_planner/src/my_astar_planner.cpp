#include "my_nav2_planner/my_astar_planner.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "nav2_util/node_utils.hpp"

#include <queue>
#include <cmath>
#include <algorithm>

// 注册插件
PLUGINLIB_EXPORT_CLASS(my_nav2_planner::MyAStarPlanner, nav2_core::GlobalPlanner)

namespace my_nav2_planner
{

// A* 节点结构
struct Node {
    int x, y;
    double g_cost;
    double h_cost;
    double f_cost;
    int parent_index;

    // 优先队列需要重载 > 运算符 (小顶堆)
    bool operator>(const Node& other) const {
        return f_cost > other.f_cost;
    }
};

MyAStarPlanner::MyAStarPlanner() : costmap_(nullptr), initialized_(false) {}

MyAStarPlanner::~MyAStarPlanner() {}

void MyAStarPlanner::configure(
  const rclcpp_lifecycle::LifecycleNode::WeakPtr & parent,
  std::string name, std::shared_ptr<tf2_ros::Buffer> /*tf*/,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  node_ = parent.lock();
  name_ = name;
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap(); // 获取原始地图指针
  global_frame_ = costmap_ros_->getGlobalFrameID();
  
  RCLCPP_INFO(node_->get_logger(), "配置 MyAStarPlanner: %s", name_.c_str());
  initialized_ = true;
}

void MyAStarPlanner::cleanup()
{
  RCLCPP_INFO(node_->get_logger(), "清理 MyAStarPlanner: %s", name_.c_str());
  costmap_ros_ = nullptr;
  costmap_ = nullptr;
  initialized_ = false;
}

void MyAStarPlanner::activate()
{
  RCLCPP_INFO(node_->get_logger(), "激活 MyAStarPlanner: %s", name_.c_str());
}

void MyAStarPlanner::deactivate()
{
  RCLCPP_INFO(node_->get_logger(), "停用 MyAStarPlanner: %s", name_.c_str());
}


nav_msgs::msg::Path MyAStarPlanner::createPlan(
  const geometry_msgs::msg::PoseStamped & start,
  const geometry_msgs::msg::PoseStamped & goal)
{
  nav_msgs::msg::Path global_path;
  global_path.header.stamp = node_->now();
  global_path.header.frame_id = global_frame_;

  if (!initialized_) {
    RCLCPP_ERROR(node_->get_logger(), "规划器未初始化");
    return global_path;
  }

  // 坐标转换 World -> Map
  unsigned int mx_start, my_start, mx_goal, my_goal;
  if (!costmap_->worldToMap(start.pose.position.x, start.pose.position.y, mx_start, my_start)) {
    RCLCPP_ERROR(node_->get_logger(), "起点在地图外");
    return global_path;
  }
  if (!costmap_->worldToMap(goal.pose.position.x, goal.pose.position.y, mx_goal, my_goal)) {
    RCLCPP_ERROR(node_->get_logger(), "终点在地图外");
    return global_path;
  }

  // 初始化数据结构
  int size_x = costmap_->getSizeInCellsX();
  int size_y = costmap_->getSizeInCellsY();
  int map_size = size_x * size_y;

  // OpenList (优先队列)
  std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open_list;
  
  // Visited 表 (记录是否访问过)
  std::vector<bool> visited(map_size, false);
  
  // Parent 表 (记录父节点索引，用于回溯，-1表示无父节点)
  std::vector<int> parent_indices(map_size, -1);

  // G Cost 表 (记录到某点的最小代价，初始化为无穷大)
  std::vector<double> g_costs(map_size, std::numeric_limits<double>::infinity());

  // 起点入队
  int start_index = costmap_->getIndex(mx_start, my_start);
  g_costs[start_index] = 0.0;
  
  Node start_node;
  start_node.x = mx_start;
  start_node.y = my_start;
  start_node.g_cost = 0.0;
  start_node.h_cost = getHeuristic(mx_start, my_start, mx_goal, my_goal);
  start_node.f_cost = start_node.g_cost + start_node.h_cost;
  start_node.parent_index = -1;

  open_list.push(start_node);

  //  A* 主循环：
    
  // 定义 4 邻域方向 (上、下、左、右)
  const int dx[4] = {0, 0, 1, -1};
  const int dy[4] = {1, -1, 0, 0};
  
  bool found_path = false;
  int goal_index = costmap_->getIndex(mx_goal, my_goal);

  while (!open_list.empty()) {
    Node current = open_list.top();
    open_list.pop();

    int current_index = costmap_->getIndex(current.x, current.y);

    // 如果已经处理过该节点则跳过
    if (visited[current_index]) continue;
    visited[current_index] = true;

    // 到达终点
    if (current_index == goal_index) {
      found_path = true;
      break;
    }

    // 扩展邻居
    for (int i = 0; i < 4; ++i) {
      int nx = current.x + dx[i];
      int ny = current.y + dy[i];

      // 边界检查
      if (nx < 0 || nx >= size_x || ny < 0 || ny >= size_y) continue;

      // 碰撞检测
      if (!isSafe(nx, ny)) continue;

      int neighbor_index = costmap_->getIndex(nx, ny);
      double new_g_cost = current.g_cost + 1.0; // 相邻移动代价为 1.0

      // 如果发现更优路径
      if (new_g_cost < g_costs[neighbor_index]) {
        g_costs[neighbor_index] = new_g_cost;
        parent_indices[neighbor_index] = current_index;

        Node neighbor;
        neighbor.x = nx;
        neighbor.y = ny;
        neighbor.g_cost = new_g_cost;
        neighbor.h_cost = getHeuristic(nx, ny, mx_goal, my_goal);
        neighbor.f_cost = new_g_cost + neighbor.h_cost;
        
        open_list.push(neighbor);
      }
    }
  }

  // 路径回溯
  if (found_path) {
    std::vector<geometry_msgs::msg::PoseStamped> path_poses;
    int curr = goal_index;

    while (curr != -1) {
      unsigned int mx, my;
      costmap_->indexToCells(curr, mx, my);
      
      double wx, wy;
      costmap_->mapToWorld(mx, my, wx, wy);

      geometry_msgs::msg::PoseStamped pose;
      pose.header = global_path.header;
      pose.pose.position.x = wx;
      pose.pose.position.y = wy;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.w = 1.0; // 简化处理，方向设为默认

      path_poses.push_back(pose);
      curr = parent_indices[curr];
    }
    
    // 反转路径 (从起点到终点)
    std::reverse(path_poses.begin(), path_poses.end());
    global_path.poses = path_poses;
    
    RCLCPP_INFO(node_->get_logger(), "成功规划路径，长度: %zu", global_path.poses.size());
  } else {
    RCLCPP_WARN(node_->get_logger(), "A* 无法找到路径");
  }

  return global_path;
}

// 启发式函数: 欧几里得距离
double MyAStarPlanner::getHeuristic(int x1, int y1, int x2, int y2) {
  return std::hypot(x2 - x1, y2 - y1);
}

// 碰撞检测
bool MyAStarPlanner::isSafe(unsigned int x, unsigned int y) {
  unsigned char cost = costmap_->getCost(x, y);
  // LETHAL_OBSTACLE = 254, INSCRIBED = 253, NO_INFORMATION = 255
  if (cost == nav2_costmap_2d::LETHAL_OBSTACLE || 
      cost == nav2_costmap_2d::INSCRIBED_INFLATED_OBSTACLE ||
      cost == nav2_costmap_2d::NO_INFORMATION) 
  {
    return false;
  }
  return true;
}

}  // namespace my_nav2_planner
