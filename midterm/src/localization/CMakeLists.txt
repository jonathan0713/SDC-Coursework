cmake_minimum_required(VERSION 3.0.2)
project(localization)

# ==========================================
# 1. 編譯器設定 (必須放在 add_executable 之前！)
# ==========================================
# 設定 C++14 標準 (ROS Noetic 基本要求)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 啟用 CPU 極致優化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# 【已修改】移除 OpenMP，因為目前的 testa 與 map_modified 程式碼不需要
# find_package(OpenMP REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")


# ==========================================
# 2. 尋找相依套件 (Find Packages)
# ==========================================
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  pcl_conversions
  pcl_ros
  tf
  tf_conversions
)

# 【已修改】移除 fast_gicp，因為目前程式碼不需要
# find_package(fast_gicp REQUIRED)

find_package(GTSAM REQUIRED)

# 必須手動尋找 PCL (你的 ICP 和 VoxelGrid 都靠它)
find_package(PCL REQUIRED)

# 如果你確實有其他檔案用到 OpenCV 才解除註解這行：
# find_package(OpenCV REQUIRED)

catkin_package()


# ==========================================
# 3. 包含標頭檔路徑 (Include Directories)
# ==========================================
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
  # ${OpenCV_INCLUDE_DIRS}
)


# ==========================================
# 4. 編譯執行檔與連結 (Build & Link)
# ==========================================
# node 1: localization
add_executable(localization src/localization.cpp)
target_link_libraries(localization 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  # ${OpenCV_LIBRARIES} 
  # fast_gicp  <-- 【已移除】
)

# node 2: map_modified
add_executable(map_modified src/map_modified.cpp)
target_link_libraries(map_modified 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
)

# node 3: visualization
add_executable(visualization src/visualization.cpp)
target_link_libraries(visualization 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES}
)

# node 4: testa
add_executable(testa src/testa.cpp)
target_link_libraries(testa 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  # ${OpenCV_LIBRARIES} 
  gtsam
  # fast_gicp  <-- 【已移除】
)

# node 5: pure_ndt
add_executable(pure_ndt src/pure_ndt.cpp)
target_link_libraries(pure_ndt 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  # ${OpenCV_LIBRARIES} 
  # fast_gicp  <-- 【已移除】
)

# node 6: testa_track1
add_executable(testa_track1 src/testa_track1.cpp)
target_link_libraries(testa_track1 
  ${catkin_LIBRARIES} 
  ${PCL_LIBRARIES} 
  # ${OpenCV_LIBRARIES} 
  gtsam
  # fast_gicp  <-- 【已移除】
)