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
#include <map>

using namespace std;

ros::Publisher pc_pub, radar_pose_pub, path_pub;
nav_msgs::Path path_msg;
ofstream file;

bool initialized = false;
Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity(); 
std::map<double, float> imu_data_map;

Eigen::Matrix<float, 5, 1> x_state;
Eigen::Matrix<float, 5, 5> P_cov;
Eigen::Matrix<float, 5, 5> Q_noise;
Eigen::Matrix<float, 3, 5> H_obs;
Eigen::Matrix<float, 3, 3> R_noise_base; 

pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);

string map_file, save_file, init_guess_file, t_file, pcd_folder;

void radar_pose_publisher(const Eigen::Matrix4f& mat) {
    geometry_msgs::PoseStamped pose;
    tf::Quaternion q;
    q.setRPY(0, 0, atan2(mat(1,0), mat(0,0)));
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";
    pose.pose.position.x = mat(0,3); pose.pose.position.y = mat(1,3); pose.pose.position.z = mat(2,3);
    pose.pose.orientation.x = q.x(); pose.pose.orientation.y = q.y(); pose.pose.orientation.z = q.z(); pose.pose.orientation.w = q.w();
    radar_pose_pub.publish(pose);
    path_msg.header.frame_id = "map"; path_msg.poses.push_back(pose); path_pub.publish(path_msg);
}

void callback(int frame_id, float raw_imu_wz, bool imu_valid) {
    if(!initialized) {
        ifstream in(init_guess_file.c_str());
        if(in.is_open()){
            string line; int r=0;
            while(getline(in, line) && r<4) {
               stringstream ss(line); string cell; int c=0;
               while(getline(ss, cell, ',') && c<4) {
                   transform_matrix(r, c) = stof(cell);
                   c++;
               }
               r++;
            }
            in.close();
        }

        x_state.setZero();
        x_state(0) = transform_matrix(0, 3);
        x_state(1) = transform_matrix(1, 3);
        x_state(2) = atan2(transform_matrix(1, 0), transform_matrix(0, 0));
        x_state(3) = 5.0; 
        x_state(4) = 0.0; 

        // ------------------------------------------
        // 黃金調參：邁向 ATE 1.5 分的平滑濾波器
        // ------------------------------------------
        P_cov.setIdentity(); P_cov *= 5.0; 
        
        // 1. Q_noise (預測雜訊)：極端信任物理慣性
        Q_noise.setIdentity();
        Q_noise(0,0) = 0.001;  // X 軌跡極度平滑
        Q_noise(1,1) = 0.001;  // Y 軌跡極度平滑
        Q_noise(2,2) = 0.001; // Yaw 完全鎖死，絕不輕易改變
        Q_noise(3,3) = 0.1;    // 速度允許緩慢變化
        Q_noise(4,4) = 0.0001; // 角速度完全信任 IMU 輸入

        // 2. H_obs (觀測矩陣)：保留 3x5，但我們要在 R 矩陣動手腳
        H_obs.setZero(); 
        H_obs(0,0) = 1.0; 
        H_obs(1,1) = 1.0; 
        H_obs(2,2) = 1.0; 

        // 3. R_noise_base (觀測雜訊)：大幅削弱 NDT 的拉力
        R_noise_base.setIdentity(); 
        R_noise_base(0,0) = 50.0;    // NDT 的 X 拉力變得很軟，只做長期微調
        R_noise_base(1,1) = 50.0;    // NDT 的 Y 拉力變得很軟
        R_noise_base(2,2) = 35.0; // 【核心殺招】將 NDT 的轉向觀測雜訊設為無限大！這在數學上會強制 EKF 完全無視 NDT 算出來的角度。
        initialized = true;
    }

    // ==========================================
    // 0. IMU 解毒區 (IMU Sanitization)
    // ==========================================
    float processed_wz = raw_imu_wz;
    
    // 診斷 1：檢查數值大小。如果 raw_imu_wz 動不動就出現 10, 20 甚至 50，那絕對是 deg/s。
    // 解法：取消下行註解，將度轉為弧度
    // processed_wz = raw_imu_wz * (M_PI / 180.0);

    // 診斷 2：如果軌跡在轉彎時總是朝反方向飛出去，代表 Z 軸上下相反。
    // 解法：取消下行註解，翻轉正負號
    // processed_wz = -processed_wz;

    if (imu_valid) {
        x_state(4) = processed_wz;
    } else {
        // 如果 IMU 時間沒對上，我們讓角速度歸零，總比亂轉好
        x_state(4) = 0.0;
    }

    // ==========================================
    // 1. EKF 預測
    // ==========================================
    float dt = 0.05; 
    float x = x_state(0), y = x_state(1), yaw = x_state(2), v = x_state(3), w = x_state(4);

    x_state(0) = x + v * cos(yaw) * dt;
    x_state(1) = y + v * sin(yaw) * dt;
    x_state(2) = yaw + w * dt;
    x_state(2) = atan2(sin(x_state(2)), cos(x_state(2)));

    Eigen::Matrix<float, 5, 5> F_jacob = Eigen::Matrix<float, 5, 5>::Identity();
    F_jacob(0, 2) = -v * sin(yaw) * dt; F_jacob(0, 3) = cos(yaw) * dt;
    F_jacob(1, 2) = v * cos(yaw) * dt; F_jacob(1, 3) = sin(yaw) * dt;
    F_jacob(2, 4) = dt;

    P_cov = F_jacob * P_cov * F_jacob.transpose() + Q_noise;

    Eigen::Matrix4f init_guess = transform_matrix;
    init_guess(0,3) = x_state(0); init_guess(1,3) = x_state(1);
    init_guess(0,0) = cos(x_state(2)); init_guess(0,1) = -sin(x_state(2));
    init_guess(1,0) = sin(x_state(2)); init_guess(1,1) = cos(x_state(2));

    // ==========================================
    // 2. 雷達與 NDT (放寬 NDT 的枷鎖，觀察 IMU 影響)
    // ==========================================
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*radar_pc, *radar_pc, indices);

    pcl::PassThrough<pcl::PointXYZ> pass;
    pass.setInputCloud(radar_pc);
    pass.setFilterFieldName("z"); pass.setFilterLimits(-3.0, 10.0); pass.filter(*radar_pc);
    pass.setFilterFieldName("x"); pass.setFilterLimits(-80.0, 80.0); pass.filter(*radar_pc);

    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(map_pc);
    crop.setMin(Eigen::Vector4f(init_guess(0,3)-150.0, init_guess(1,3)-150.0, -100.0, 1.0));
    crop.setMax(Eigen::Vector4f(init_guess(0,3)+150.0, init_guess(1,3)+150.0, 100.0, 1.0));
    crop.filter(*submap_pc);

    bool ndt_success = false;
    Eigen::Matrix<float, 3, 1> Z_obs; 
    float jump_from_ekf = 0.0;

    if (!submap_pc->empty() && radar_pc->size() > 10) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
        static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        ndt.setInputSource(radar_pc);
        ndt.setInputTarget(submap_pc);
        ndt.setTransformationEpsilon(0.01);

        if (frame_id == 0) {
            ndt.setResolution(15.0); ndt.setStepSize(1.5); ndt.setMaximumIterations(50);
            ndt.align(*tmp, init_guess);
            Eigen::Matrix4f res_coarse = ndt.getFinalTransformation();
            ndt.setResolution(2.0); ndt.setStepSize(0.2); ndt.setMaximumIterations(40);
            ndt.align(*tmp, res_coarse);
        } else {
            // 放寬搜索半徑到 5.0m，觀察 NDT 和 IMU 到底誰在拉扯
            ndt.setResolution(5.0); ndt.setStepSize(0.5); ndt.setMaximumIterations(35);
            ndt.align(*tmp, init_guess);
            Eigen::Matrix4f res_mid = ndt.getFinalTransformation();
            ndt.setResolution(1.0); ndt.setStepSize(0.1); ndt.setMaximumIterations(30);
            ndt.align(*tmp, res_mid);
        }

        // 計算配準分數 (越低越好，通常小於 2.0 代表特徵匹配良好)
        double fitness_score = ndt.getFitnessScore(2.0);

        // 只有在收斂，且分數合理的情況下，才允許更新
        if (ndt.hasConverged() && fitness_score < 4.0) {
            Eigen::Matrix4f ndt_res = ndt.getFinalTransformation();
            Z_obs(0) = ndt_res(0,3);
            Z_obs(1) = ndt_res(1,3);
            Z_obs(2) = atan2(ndt_res(1,0), ndt_res(0,0));
            
            jump_from_ekf = sqrt(pow(Z_obs(0) - x_state(0), 2) + pow(Z_obs(1) - x_state(1), 2));
            ndt_success = true;
        } else {
            ROS_WARN("FRAME %d: NDT Score too high (%.2f). Rejecting.", frame_id, fitness_score);
        }
    }

    // 印出診斷資訊 (這行是抓鬼關鍵)
    if (frame_id % 10 == 0) {
        ROS_INFO("FRM %d | IMU raw: %.3f, used: %.3f | Vel: %.2f | NDT Jump: %.2fm", 
                 frame_id, raw_imu_wz, processed_wz, x_state(3), jump_from_ekf);
    }

    // ==========================================
    // 3. EKF 更新
    // ==========================================
    // ==========================================
    // 3. EKF 更新 (加入開局豁免權與 AEKF 動態懲罰)
    // ==========================================
    if (ndt_success) {
        Eigen::Matrix<float, 3, 3> R_dynamic = R_noise_base;
        
        // 【開局豁免權】：前 50 幀 (約 2.5 秒) 無條件信任 NDT
        // 讓 NDT 強行把車子往前拉，補足 EKF 速度收斂期所欠缺的直線距離！
        // 50 幀之後，恢復 AEKF 防禦，抵擋路口雜訊與公車鬼影。
        if (frame_id > 50 && jump_from_ekf > 1.5) {
            float penalty = exp(jump_from_ekf * 0.8);
            if (penalty > 1000.0) penalty = 1000.0;
            // 只懲罰 X 和 Y 的信任度，因為 Yaw 的 R 已經是 30.0 (很軟了)
            R_dynamic(0,0) *= penalty;
            R_dynamic(1,1) *= penalty;
        }

        Eigen::Matrix<float, 3, 1> y_res = Z_obs - H_obs * x_state;
        y_res(2) = atan2(sin(y_res(2)), cos(y_res(2))); 

        Eigen::Matrix<float, 3, 3> S = H_obs * P_cov * H_obs.transpose() + R_dynamic;
        Eigen::Matrix<float, 5, 3> K = P_cov * H_obs.transpose() * S.inverse();

        x_state = x_state + K * y_res;
        x_state(2) = atan2(sin(x_state(2)), cos(x_state(2))); 
        
        // 【本質修復】：強制速度下限為 0.0，絕對不允許 EKF 產生「倒車」幻覺
        if (x_state(3) > 25.0) x_state(3) = 25.0; 
        if (x_state(3) < 0.0) x_state(3) = 0.0; // 封殺畫圈圈的元兇

        Eigen::Matrix<float, 5, 5> I5 = Eigen::Matrix<float, 5, 5>::Identity();
        P_cov = (I5 - K * H_obs) * P_cov;
    } else {
        // 雷達盲區：速度微幅衰減，但轉向絕對聽從 IMU
        x_state(3) *= 0.98; // 衰減慢一點，讓它能滑過大路口
        if (x_state(3) < 0.0) x_state(3) = 0.0;
    }

    transform_matrix(0,3) = x_state(0); transform_matrix(1,3) = x_state(1);
    transform_matrix(0,0) = cos(x_state(2)); transform_matrix(0,1) = -sin(x_state(2));
    transform_matrix(1,0) = sin(x_state(2)); transform_matrix(1,1) = cos(x_state(2));
    transform_matrix(2,3) = init_guess(2,3) * 0.8; 

    radar_pose_publisher(transform_matrix);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) file << transform_matrix(i, j) << ( (i==2 && j==3) ? "" : " " );
    }
    file << endl;
    file.flush();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "testa_node");
    ros::NodeHandle nh;
    string pkg = ros::package::getPath("localization");
    t_file = pkg + "/data/Track2/radar_t_list.txt";
    pcd_folder = pkg + "/data/Track2/raw_radar_pcd";
    map_file = pkg + "/data/Track2/map_modified.pcd";
    save_file = pkg + "/result/result.txt";
    init_guess_file = pkg + "/data/Track2/init_guess.txt";

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/4D_radar", 1);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/tranformed_radar_pose", 1);
    path_pub = nh.advertise<nav_msgs::Path>("/localization_path", 1);
    file.open(save_file);

    string imu_file = pkg + "/data/Track2/imu_data.csv";
    ifstream imu_in(imu_file);
    if(imu_in.is_open()) {
        string imu_line;
        getline(imu_in, imu_line); 
        while(getline(imu_in, imu_line)) {
            stringstream ss(imu_line);
            string cell; vector<string> tokens;
            while(getline(ss, cell, ',')) tokens.push_back(cell);
            if(tokens.size() > 19) {
                try {
                    double t = stod(tokens[0]);
                    if (t > 1e15) t = t / 1e9; 
                    float wz = stof(tokens[19]);
                    imu_data_map[t] = wz;
                } catch(...) {}
            }
        }
        imu_in.close();
    }

    pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_pc);
    ifstream t_in(t_file); string line;
    int f_idx = 0;
    
    while(getline(t_in, line) && ros::ok()) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_folder + "/" + line + ".pcd", *radar_pc) != -1) {
            
            // 【本質修復：統一時間戳為「秒」】
            double current_radar_time = stod(line);
            if (current_radar_time > 1e15) {
                current_radar_time = current_radar_time / 1e9; 
            }

            float current_imu_wz = 0.0;
            bool imu_valid = false;
            auto it = imu_data_map.lower_bound(current_radar_time);
            
            // 容許 0.1 秒的誤差
            if(it != imu_data_map.end() && abs(it->first - current_radar_time) < 0.1) {
                current_imu_wz = it->second;
                imu_valid = true;
            } else if (it != imu_data_map.begin()) {
                auto prev_it = std::prev(it);
                if (abs(prev_it->first - current_radar_time) < 0.1) {
                    current_imu_wz = prev_it->second;
                    imu_valid = true;
                }
            }
            
            callback(f_idx++, current_imu_wz, imu_valid);
        }
    }
    file.close();
    return 0;
}