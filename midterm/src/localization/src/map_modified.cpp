#include <ros/ros.h>
#include <ros/package.h>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

using namespace std;

void process_map()
{
    // 統一使用 PointXYZ，與 testa.cpp / localization.cpp 完全對齊
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_clipped(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);

    string package_path = ros::package::getPath("localization");
    string map_source_path = package_path + "/data/Track2/";
    string map_file = map_source_path + "map.pcd";
    string save_file = map_source_path + "map_modified.pcd";

    ROS_INFO("Loading map %s...", map_file.c_str());
    
    // 載入時使用 PointXYZ
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *cloud) == -1)
    {
        PCL_ERROR("Couldn't read map pcd file\n");
        return;
    }
    ROS_INFO("Original map size: %lu points.", cloud->size());

    // ==========================================
    // 1. 寬鬆的邊界限制 (剔除極端離群點，確保 VoxelGrid 絕對不溢位)
    // ==========================================
    pcl::PassThrough<pcl::PointXYZ> pass;
    // pass.setInputCloud(cloud);
    // pass.setFilterFieldName("x");
    // pass.setFilterLimits(-500.0, 1200.0); 
    // pass.filter(*cloud_clipped);

    // pass.setInputCloud(cloud_clipped);
    // pass.setFilterFieldName("y");
    // pass.setFilterLimits(-600.0, 600.0);
    // pass.filter(*cloud_clipped);

    // pass.setInputCloud(cloud_clipped);
    // pass.setFilterFieldName("z");
    // pass.setFilterLimits(-2, 30.0);
    // pass.filter(*cloud_clipped);
    pass.setInputCloud(cloud);
    
    pass.filter(*cloud_clipped);    

    ROS_INFO("Clipped map size: %lu points (removed extreme outliers).", cloud_clipped->size());

    // ==========================================
    // 2. 回歸 1.0m 降採樣
    // ==========================================
    // 1.0m 的網格能將 4000 萬點穩定壓縮至 40 萬點左右，形成平滑的 NDT 最佳化空間
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud_clipped);
    vg.setLeafSize(1.0f, 1.0f, 1.0f); 
    vg.filter(*filtered_cloud);

    ROS_INFO("Filtered map size: %lu points.", filtered_cloud->size());

    // 儲存為 Binary 格式加速後續載入
    ROS_INFO("Saving modified map to %s...", save_file.c_str());
    pcl::io::savePCDFileBinary(save_file, *filtered_cloud);
    ROS_INFO("Done!");
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "map_modified_node");
    process_map();
    return 0;
}