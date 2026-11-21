#include <string>
#include <fstream>
#include <iostream>
#include <cmath>
#include <vector>
#include <exception>

#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
// 【新增】引入 Marker 標頭檔
#include <visualization_msgs/Marker.h> 

using namespace std;

ros::Publisher path_pub;
ros::Publisher radar_pose_pub;
nav_msgs::Path path;

// 將 4x4 轉換矩陣轉為 TF
void matrix4fToTfTransform(const Eigen::Matrix4f &matrix, tf::Transform &tf_transform) {
    tf::Matrix3x3 tf_rotation(
        matrix(0, 0), matrix(0, 1), matrix(0, 2),
        matrix(1, 0), matrix(1, 1), matrix(1, 2),
        matrix(2, 0), matrix(2, 1), matrix(2, 2)
    );
    tf::Vector3 tf_translation(matrix(0, 3), matrix(1, 3), matrix(2, 3));
    tf_transform.setBasis(tf_rotation);
    tf_transform.setOrigin(tf_translation);
    tf::Quaternion q = tf_transform.getRotation();
    q.normalize();
    tf_transform.setRotation(q);
}

// 發布雷達的 Pose (姿態) 與 Path (軌跡)
void radar_pose_publisher(float x, float y, float z, float yaw)
{
    geometry_msgs::PoseStamped pose;
    tf2::Quaternion myQuaternion;
    myQuaternion.setRPY(0, 0, yaw);
    myQuaternion.normalize();

    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";

    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = z;

    pose.pose.orientation.x = myQuaternion.getX();
    pose.pose.orientation.y = myQuaternion.getY();
    pose.pose.orientation.z = myQuaternion.getZ();
    pose.pose.orientation.w = myQuaternion.getW();
    
    radar_pose_pub.publish(pose);
    
    path.header.frame_id = "map";
    path.poses.push_back(pose);
    path_pub.publish(path);
}


int main(int argc, char** argv)
{
    ros::init (argc, argv, "map_visualization");
    ros::NodeHandle nh;

    string package_path = ros::package::getPath("localization");
    if (package_path.empty()) {
        ROS_ERROR("Failed to find localization package path.");
        return -1;
    }

    ros::Publisher map_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/map", 1, true);
    ros::Publisher radar_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/radar", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/visual/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/visual/localization_path", 1);
    
    // 【新增】宣告 Text Marker 的 Publisher
    ros::Publisher text_marker_pub = nh.advertise<visualization_msgs::Marker>("/visual/frame_text", 1);

    tf::TransformBroadcaster br;

    string map_file = package_path + "/data/Track2/map_modified.pcd";
    string t_file = package_path + "/data/Track2/radar_t_list.txt";  
    string pcd_folder = package_path + "/data/Track2/raw_radar_pcd/";
    string pose_file = package_path + "/result/result.txt";

    // 1. 讀取並發布地圖 (修正型別為 PointXYZ，避免 Intensity 報錯)
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
    ROS_INFO("Loading global map for visualization...");
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_pc) == -1)
    {
        PCL_ERROR ("Couldn't read map file\n");
        return -1;
    }

    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*map_pc, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    map_pub.publish(map_msg);
    ROS_INFO("Map published!");

    // 2. 讀取 Timestamp 列表與定位結果檔
    vector<string> t_list;
    ifstream t_in(t_file);
    ifstream pose_in(pose_file);

    if(!t_in.is_open() || !pose_in.is_open())
    {
        ROS_ERROR("Error opening radar list or result.txt");
        return -1;
    }

    string line;
    while(getline(t_in, line) && ros::ok())
    {
        if (!line.empty() && line[line.length()-1] == '\r') line.erase(line.length()-1);
        string pcd_file = pcd_folder + line + ".pcd";
        t_list.push_back(pcd_file);
    }  

    int idx = 0;
    ROS_INFO("Starting playback...");

    // 3. 迴圈播放每一幀雷達與軌跡
    while(getline(pose_in, line) && ros::ok() && idx < t_list.size())
    {
        if (line.empty()) continue;

        stringstream ss(line);
        string s;
        vector<string> record;
        
        while(getline(ss, s, ' '))
        {
            if(!s.empty() && s != "\r") record.push_back(s);
        }

        if (record.size() < 12) {
            ROS_WARN("Invalid pose format at line %d, skipping.", idx);
            idx++;
            continue;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(t_list[idx], *radar_pc) == -1)
        {
            ROS_ERROR ("Couldn't read radar file: %s", t_list[idx].c_str());
            idx++;
            continue;
        }

        // ==========================================
        // 防呆崩潰區塊：用 try-catch 攔截爆掉的浮點數
        // ==========================================
        Eigen::Matrix4f transform = Eigen::Matrix4f::Identity();
        try {
            // 改用 stod (String to Double)，容受度從 10^38 提升到 10^308
            transform << stod(record[0]), stod(record[1]), stod(record[2]), stod(record[3]), 
                         stod(record[4]), stod(record[5]), stod(record[6]), stod(record[7]), 
                         stod(record[8]), stod(record[9]), stod(record[10]), stod(record[11]),
                         0, 0, 0, 1;
        } catch (const std::exception& e) {
            ROS_ERROR("CRASH PREVENTED at frame %d! Invalid number format: %s", idx, e.what());
            ROS_ERROR("Bad Line Content: %s", line.c_str());
            idx++;
            continue; // 跳過這個壞掉的幀，不要當機
        }

        // 發布 TF
        tf::Transform tf_transform;
        matrix4fToTfTransform(transform, tf_transform);
        br.sendTransform(tf::StampedTransform(tf_transform, ros::Time::now(), "map", "radar"));                     
        
        // 發布點雲
        sensor_msgs::PointCloud2 radar_msg;
        pcl::toROSMsg(*radar_pc, radar_msg);
        radar_msg.header.frame_id = "radar";
        radar_msg.header.stamp = ros::Time::now();
        radar_pub.publish(radar_msg);

        // 發布軌跡
        radar_pose_publisher(transform(0, 3), transform(1, 3), transform(2, 3), atan2(transform(1, 0), transform(0, 0)));

        // ==========================================
        // 【新增】1. 終端機進度輸出 (每 10 幀印一次避免洗畫面)
        // ==========================================
        if (idx % 10 == 0) {
            ROS_INFO("Playback Progress: Frame %d / %lu", idx, t_list.size());
        }

        // ==========================================
        // 【新增】2. RViz 漂浮文字標籤 (Text Marker)
        // ==========================================
        visualization_msgs::Marker text_marker;
        text_marker.header.frame_id = "map";
        text_marker.header.stamp = ros::Time::now();
        text_marker.ns = "frame_info";
        text_marker.id = 0;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING; // 文字會永遠面向視角
        text_marker.action = visualization_msgs::Marker::ADD;

        // 將文字設定在車子上方 5 公尺處 (避免被點雲擋住)
        text_marker.pose.position.x = transform(0, 3);
        text_marker.pose.position.y = transform(1, 3);
        text_marker.pose.position.z = transform(2, 3) + 5.0; 
        text_marker.pose.orientation.w = 1.0;

        // 設定字體大小與顏色 (黃色)
        text_marker.scale.z = 3.0; // 字體高度 3 公尺
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 0.0;
        text_marker.color.a = 1.0; // 透明度 (1.0 = 不透明)

        // 設定要顯示的文字內容
        char text_buf[256];
        sprintf(text_buf, "Frame: %d", idx);
        text_marker.text = text_buf;

        // 發布 Marker
        text_marker_pub.publish(text_marker);
        // ==========================================

        idx++;
        ros::Duration(0.05).sleep(); 
    }

    ROS_INFO("Playback finished. Keep spinning for RViz.");
    ros::spin(); 
    return 0;
}