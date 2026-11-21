#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/ndt.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <vector>

using namespace std;

// 全域 Publisher 與變數
ros::Publisher pc_pub, radar_pose_pub, path_pub, map_pub;
nav_msgs::Path path_msg;
ofstream file;
Eigen::Matrix4f current_pose = Eigen::Matrix4f::Identity(); 

// tf 轉換副程式
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

void radar_pose_publisher(const Eigen::Matrix4f& mat) {
    geometry_msgs::PoseStamped pose;
    tf::Quaternion q;
    q.setRPY(0, 0, atan2(mat(1,0), mat(0,0)));
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";
    pose.pose.position.x = mat(0,3); 
    pose.pose.position.y = mat(1,3); 
    pose.pose.position.z = mat(2,3);
    pose.pose.orientation.x = q.x(); 
    pose.pose.orientation.y = q.y(); 
    pose.pose.orientation.z = q.z(); 
    pose.pose.orientation.w = q.w();
    
    radar_pose_pub.publish(pose);
    path_msg.header.frame_id = "map"; 
    path_msg.poses.push_back(pose); 
    path_pub.publish(path_msg);

    static tf::TransformBroadcaster br;
    tf::Transform tf_transform;
    matrix4fToTfTransform(mat, tf_transform);
    br.sendTransform(tf::StampedTransform(tf_transform, ros::Time::now(), "map", "radar"));
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "pure_ndt_node");
    ros::NodeHandle nh;
    
    string pkg = ros::package::getPath("localization");
    string t_file = pkg + "/data/Track2/radar_t_list.txt";
    string pcd_folder = pkg + "/data/Track2/raw_radar_pcd";
    string map_file = pkg + "/data/Track2/map_modified.pcd"; // 確保這是 0.2 或 0.5m 降採樣的高精度地圖
    string save_file = pkg + "/result/pure_ndt_result.txt";
    string init_guess_file = pkg + "/data/Track2/init_guess.txt";

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/radar", 10);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/visual/tranformed_radar_pose", 10);
    path_pub = nh.advertise<nav_msgs::Path>("/visual/localization_path", 10);
    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/map", 1, true);

    file.open(save_file);

    // 1. 載入並發布地圖
    pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc(new pcl::PointCloud<pcl::PointXYZ>);
    ROS_INFO("Loading Global Map...");
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_pc) == -1) {
        ROS_ERROR("Failed to load map.");
        return -1;
    }
    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*map_pc, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    map_pub.publish(map_msg);

    // 2. 載入 Initial Guess (第 0 幀的起點)
    ifstream in(init_guess_file.c_str());
    if(in.is_open()){
        string line; int r=0;
        while(getline(in, line) && r<4) {
            stringstream ss(line); string cell; int c=0;
            while(getline(ss, cell, ',') && c<4) {
                current_pose(r, c) = stof(cell);
                c++;
            }
            r++;
        }
        in.close();
        ROS_INFO("Initial Guess Loaded.");
    }

    // 迴圈與預測模型變數
    ifstream t_in(t_file); 
    string line;
    int frame_id = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc(new pcl::PointCloud<pcl::PointXYZ>);
    
    // 用於恆定速度預測 (Constant Velocity Model)
    Eigen::Matrix4f last_pose = current_pose;
    bool is_first_frame = true;
    double initial_z_height = current_pose(2,3); // 記住起始高度，永遠鎖死在這個高度

    ROS_INFO("Starting Pure NDT Loop with CV Model...");
    
    while(getline(t_in, line) && ros::ok()) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_folder + "/" + line + ".pcd", *radar_pc) != -1) {
            
            // ==========================================
            // Step A: 雷達點雲前處理 (找回車道邊界特徵)
            // ==========================================
            std::vector<int> indices;
            pcl::removeNaNFromPointCloud(*radar_pc, *radar_pc, indices);

            pcl::PassThrough<pcl::PointXYZ> pass;
            pass.setInputCloud(radar_pc);
            // 範圍設定為 0.2m ~ 3.0m，保留路緣石、護欄與車身，避開地面與過高樹冠
            pass.setFilterFieldName("z"); 
            // pass.setFilterLimits(-5.0, 15.0); 
            pass.filter(*radar_pc);
            
            pass.setFilterFieldName("x"); pass.setFilterLimits(-80.0, 80.0); pass.filter(*radar_pc);
            pass.setFilterFieldName("y"); pass.setFilterLimits(-40.0, 40.0); pass.filter(*radar_pc);

            // ==========================================
            // 新增：恆定速度預測 (CV Prediction)
            // ==========================================
            Eigen::Matrix4f initial_guess = current_pose;
            if (!is_first_frame) {
                // Delta = (上一幀)^-1 * (當前幀)
                Eigen::Matrix4f delta_transform = last_pose.inverse() * current_pose;
                // 將這個 Delta 繼續乘上去，預測下一幀的落點
                initial_guess = current_pose * delta_transform;
            }
            last_pose = current_pose; // 更新歷史紀錄
            is_first_frame = false;

            // ==========================================
            // Step B: 局部地圖裁切 (強勢壓扁 Z 軸)
            // ==========================================
            pcl::CropBox<pcl::PointXYZ> crop;
            crop.setInputCloud(map_pc);
            
            float cx = initial_guess(0,3);
            float cy = initial_guess(1,3);
            // 地圖的 Z 軸也必須嚴格對齊雷達的 0.2m ~ 3.0m
            crop.setMin(Eigen::Vector4f(cx - 100.0, cy - 60.0, -5.0, 1.0));
            crop.setMax(Eigen::Vector4f(cx + 100.0, cy + 60.0, 15.0, 1.0));
            crop.filter(*submap_pc);

            // ==========================================
            // Step C: 純 NDT 兩段式配準 (Coarse-to-Fine)
            // ==========================================
            pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
            
            ndt.setInputSource(radar_pc);
            ndt.setInputTarget(submap_pc);
            ndt.setOulierRatio(0.45); 

            // 1. 粗配準 (引力範圍大，處理預測誤差)
            ndt.setResolution(3.0);              
            ndt.setStepSize(0.5);                
            ndt.setTransformationEpsilon(0.01);  
            ndt.setMaximumIterations(35);
            ndt.align(*tmp, initial_guess); // 投入我們用速度預測出來的 Initial Guess
            Eigen::Matrix4f coarse_pose = ndt.getFinalTransformation();

            // 2. 細配準 (鎖定公分級精度)
            ndt.setResolution(0.5);              
            ndt.setStepSize(0.1);                
            ndt.setTransformationEpsilon(5e-5);  
            ndt.setMaximumIterations(50);
            ndt.align(*tmp, coarse_pose);

            // ==========================================
            // Step D: 結果驗證與鎖定 2.5D 自由度
            // ==========================================
            bool has_converged = ndt.hasConverged();
            double fitness_score = ndt.getFitnessScore(1.0); 
            
            if (has_converged && fitness_score < 1.0) {
                Eigen::Matrix4f ndt_result = ndt.getFinalTransformation();
                
                // 提取 Roll, Pitch, Yaw
                tf::Matrix3x3 mat(
                    ndt_result(0,0), ndt_result(0,1), ndt_result(0,2),
                    ndt_result(1,0), ndt_result(1,1), ndt_result(1,2),
                    ndt_result(2,0), ndt_result(2,1), ndt_result(2,2)
                );
                double roll, pitch, yaw;
                mat.getRPY(roll, pitch, yaw);
                
                // 重組矩陣：只保留 Yaw 角，強制 Roll 和 Pitch 為 0
                current_pose(0,0) = cos(yaw);  current_pose(0,1) = -sin(yaw); current_pose(0,2) = 0.0;
                current_pose(1,0) = sin(yaw);  current_pose(1,1) = cos(yaw);  current_pose(1,2) = 0.0;
                current_pose(2,0) = 0.0;       current_pose(2,1) = 0.0;       current_pose(2,2) = 1.0;
                
                // 只更新 X 與 Y，強制鎖死 Z 的高度
                current_pose(0,3) = ndt_result(0,3);
                current_pose(1,3) = ndt_result(1,3);
                current_pose(2,3) = ndt_result(2,3);

                ROS_INFO("FRM %d | Score: %.3f | Iters: %d | X: %.2f, Y: %.2f", 
                          frame_id, fitness_score, ndt.getFinalNumIteration(), current_pose(0,3), current_pose(1,3));
            } else {
                // 如果配準失敗（例如經過完全沒特徵的空曠區塊）
                // 就直接沿用 CV Model 預測的 initial_guess 往前滑行，撐過盲區！
                current_pose = initial_guess;
                // 確保高度一樣鎖死
                current_pose(2,3) = initial_z_height; 
                ROS_WARN("FRM %d | NDT Failed/Bad Score (%.3f). Coasting with CV Model...", frame_id, fitness_score);
            }

            // 發布資料與儲存
            radar_pose_publisher(current_pose);
            
            // 將原始點雲透過剛剛算出的姿態進行轉換，方便 RViz 觀察疊合狀況
            pcl::transformPointCloud(*radar_pc, *tmp, current_pose);
            sensor_msgs::PointCloud2 radar_msg;
            pcl::toROSMsg(*tmp, radar_msg); 
            radar_msg.header.frame_id = "map"; 
            radar_msg.header.stamp = ros::Time::now();
            pc_pub.publish(radar_msg);

            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 4; ++j) {
                    file << current_pose(i, j) << ( (i==2 && j==3) ? "" : " " );
                }
            }
            file << endl;
            file.flush();

            frame_id++;
            ros::spinOnce();
            ros::Duration(0.01).sleep(); 
        }
    }
    
    file.close();
    ROS_INFO("Pure NDT with CV Model Processing finished.");
    ros::spin(); 
    return 0;
}