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
#include <iomanip>
#include <vector>

using namespace std;

// ==========================================
// Checkpoint 控制面板
// ==========================================
bool LOAD_CHECKPOINT = false;         
bool SAVE_CHECKPOINT = true;          
int TARGET_CHECKPOINT_FRAME = 1000;   
bool checkpoint_loaded = false;       

// 全域 Publisher 與變數宣告
ros::Publisher pc_pub, radar_pose_pub, path_pub, map_pub;
nav_msgs::Path path_msg;
ofstream file;

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

bool initialized = false;
Eigen::Matrix4f transform_matrix = Eigen::Matrix4f::Identity(); 
int blind_exit_timer = 0; 

struct ImuData {
    float wz;
    float ax;
};
std::map<double, ImuData> imu_data_map;

Eigen::Matrix<float, 5, 1> x_state;
Eigen::Matrix<float, 5, 5> P_cov;
Eigen::Matrix<float, 5, 5> Q_noise;
Eigen::Matrix<float, 3, 5> H_obs;
Eigen::Matrix<float, 3, 3> R_noise_base; 

pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);

string map_file, save_file, init_guess_file, t_file, pcd_folder, checkpoint_file; 

void radar_pose_publisher(const Eigen::Matrix4f& mat) {
    geometry_msgs::PoseStamped pose;
    tf::Quaternion q;
    q.setRPY(0, 0, atan2(mat(1,0), mat(0,0)));
    pose.header.stamp = ros::Time::now();
    pose.header.frame_id = "map";
    pose.pose.position.x = mat(0,3); pose.pose.position.y = mat(1,3); pose.pose.position.z = mat(2,3);
    pose.pose.orientation.x = q.x(); pose.pose.orientation.y = q.y(); pose.pose.orientation.z = q.z(); pose.pose.orientation.w = q.w();
    
    radar_pose_pub.publish(pose);
    path_msg.header.frame_id = "map"; 
    path_msg.poses.push_back(pose); 
    path_pub.publish(path_msg);

    static tf::TransformBroadcaster br;
    tf::Transform tf_transform;
    matrix4fToTfTransform(mat, tf_transform);
    br.sendTransform(tf::StampedTransform(tf_transform, ros::Time::now(), "map", "radar"));
}

void callback(int frame_id, const std::vector<std::pair<double, ImuData>>& imu_seq, double last_radar_time, double current_radar_time) {
    if(!initialized) {
        if (!checkpoint_loaded) {
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
            x_state(2) = 0.5;
            x_state(3) = 5.5; 
            x_state(4) = 0.0; 

            P_cov.setZero(); 
            P_cov(0,0) = 1.0; P_cov(1,1) = 1.0; P_cov(2,2) = 0.1; P_cov(3,3) = 0.6; P_cov(4,4) = 0.6; 
        }
        
        Q_noise.setIdentity();
        Q_noise(0,0) = 0.001;  
        Q_noise(1,1) = 0.001;  
        Q_noise(2,2) = 0.0025; 
        Q_noise(3,3) = 0.2;    
        Q_noise(4,4) = 0.0004; 

        H_obs.setZero(); 
        H_obs(0,0) = 1.0; H_obs(1,1) = 1.0; H_obs(2,2) = 1.0; 

        R_noise_base.setIdentity(); 
        R_noise_base(0,0) = 45; R_noise_base(1,1) = 45; R_noise_base(2,2) = 30; 
        initialized = true;
    }

    // ==========================================
    // 1. EKF 預測 (100Hz 內部高頻積分)
    // ==========================================
    float total_dt_for_clamp = 0.05; 

    if (last_radar_time <= 0.0 || imu_seq.empty()) {
        float dt = current_radar_time > last_radar_time ? (current_radar_time - last_radar_time) : 0.05;
        if (dt > 0.2) dt = 0.05;
        total_dt_for_clamp = dt;

        float x = x_state(0), y = x_state(1), yaw = x_state(2), v = x_state(3), w = x_state(4);
        x_state(0) = x + v * dt * cos(yaw); x_state(1) = y + v * dt * sin(yaw); x_state(2) = yaw + w * dt;
        x_state(2) = atan2(sin(x_state(2)), cos(x_state(2)));

        Eigen::Matrix<float, 5, 5> F_jacob = Eigen::Matrix<float, 5, 5>::Identity();
        F_jacob(0, 2) = -v * dt * sin(yaw); F_jacob(0, 3) = cos(yaw) * dt;
        F_jacob(1, 2) = v * dt * cos(yaw);  F_jacob(1, 3) = sin(yaw) * dt; F_jacob(2, 4) = dt;
        P_cov = F_jacob * P_cov * F_jacob.transpose() + Q_noise;
        
    } else {
        double prev_time = last_radar_time;
        for (size_t i = 0; i < imu_seq.size(); ++i) {
            double current_t = imu_seq[i].first;
            float raw_wz = imu_seq[i].second.wz;
            float raw_ax = imu_seq[i].second.ax;
            
            float dt = current_t - prev_time;
            if (dt <= 0.001) continue; 
            // if (dt > 0.1) dt = 0.01;   
            
            x_state(4) = raw_wz; 
            
            if (abs(raw_ax) > 0.0) { 
                x_state(3) = x_state(3) + raw_ax * dt; 
            }
            if (x_state(3) < 0.0) x_state(3) = 0.0;
            
            float x = x_state(0), y = x_state(1), yaw = x_state(2), v = x_state(3), w = x_state(4);
            x_state(0) = x + v * dt * cos(yaw);
            x_state(1) = y + v * dt * sin(yaw);
            x_state(2) = yaw + w * dt;
            x_state(2) = atan2(sin(x_state(2)), cos(x_state(2)));

            Eigen::Matrix<float, 5, 5> F_jacob = Eigen::Matrix<float, 5, 5>::Identity();
            F_jacob(0, 2) = -v * dt * sin(yaw); F_jacob(0, 3) = cos(yaw) * dt;
            F_jacob(1, 2) = v * dt * cos(yaw);  F_jacob(1, 3) = sin(yaw) * dt; F_jacob(2, 4) = dt;

            Eigen::Matrix<float, 5, 5> Q_scaled = Q_noise * (dt / 0.05f);
            P_cov = F_jacob * P_cov * F_jacob.transpose() + Q_scaled;

            prev_time = current_t;
        }

        float dt_remainder = current_radar_time - prev_time;
        if (dt_remainder > 0.001 && dt_remainder < 0.1) {
            float dt = dt_remainder;
            float x = x_state(0), y = x_state(1), yaw = x_state(2), v = x_state(3), w = x_state(4);
            x_state(0) = x + v * dt * cos(yaw); x_state(1) = y + v * dt * sin(yaw); x_state(2) = yaw + w * dt;
            x_state(2) = atan2(sin(x_state(2)), cos(x_state(2)));

            Eigen::Matrix<float, 5, 5> F_jacob = Eigen::Matrix<float, 5, 5>::Identity();
            F_jacob(0, 2) = -v * dt * sin(yaw); F_jacob(0, 3) = cos(yaw) * dt;
            F_jacob(1, 2) = v * dt * cos(yaw);  F_jacob(1, 3) = sin(yaw) * dt; F_jacob(2, 4) = dt;
            P_cov = F_jacob * P_cov * F_jacob.transpose() + Q_noise * (dt / 0.05f);
        }
        total_dt_for_clamp = current_radar_time - last_radar_time;
    }

    // 狀態機：大彎與盲區判定
    float wz_mag = abs(x_state(4)); 
    bool is_hard_turning = (wz_mag > 0.20); 
    if (is_hard_turning) blind_exit_timer = 60; 
    else if (blind_exit_timer > 0) blind_exit_timer--;
    bool is_blind_exit = (blind_exit_timer > 0);

    Eigen::Matrix4f init_guess = transform_matrix;
    init_guess(0,3) = x_state(0); init_guess(1,3) = x_state(1);
    init_guess(0,0) = cos(x_state(2)); init_guess(0,1) = -sin(x_state(2));
    init_guess(1,0) = sin(x_state(2)); init_guess(1,1) = cos(x_state(2));

    // ==========================================
    // 2. 雷達與 NDT
    // ==========================================
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*radar_pc, *radar_pc, indices);

    if(frame_id > 1000){
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(radar_pc);
        pass.setFilterFieldName("z"); pass.setFilterLimits(-3.0, 10.0); pass.filter(*radar_pc);
        pass.setFilterFieldName("x"); pass.setFilterLimits(-100.0, 100.0); pass.filter(*radar_pc);
        pass.setFilterFieldName("y"); pass.setFilterLimits(-60.0, 60.0); pass.filter(*radar_pc);
    } else {
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(radar_pc);
        pass.setFilterFieldName("z"); pass.setFilterLimits(-3.0, 10.0); pass.filter(*radar_pc);
        pass.setFilterFieldName("x"); pass.setFilterLimits(-80.0, 80.0); pass.filter(*radar_pc);
        pass.setFilterFieldName("y"); pass.setFilterLimits(-40.0, 40.0); pass.filter(*radar_pc);
    }
    
    if (frame_id < 1105) {
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(map_pc);
        crop.setMin(Eigen::Vector4f(init_guess(0,3)-100.0, init_guess(1,3)-60.0, -100.0, 1.0));
        crop.setMax(Eigen::Vector4f(init_guess(0,3)+100.0, init_guess(1,3)+60.0, 100.0, 1.0));
        crop.filter(*submap_pc);
    } else {
        pcl::CropBox<pcl::PointXYZ> crop;
        crop.setInputCloud(map_pc);
        crop.setMin(Eigen::Vector4f(init_guess(0,3)-100.0, init_guess(1,3)-60.0, -100.0, 1.0));
        crop.setMax(Eigen::Vector4f(init_guess(0,3)+100.0, init_guess(1,3)+60.0, 100.0, 1.0));
        crop.filter(*submap_pc);
    }   

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
            ndt.setResolution(5.0); ndt.setStepSize(0.5); ndt.setMaximumIterations(35);
            ndt.align(*tmp, init_guess);
            Eigen::Matrix4f res_mid = ndt.getFinalTransformation();
            ndt.setResolution(1.0); ndt.setStepSize(0.1); ndt.setMaximumIterations(30);
            ndt.align(*tmp, res_mid);
        }

        double fitness_score = ndt.getFitnessScore(2.0);

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

    if (frame_id % 10 == 0) {
        ROS_INFO("FRM %d | v: %.2f | NDT Jump: %.2fm", frame_id, x_state(3), jump_from_ekf);
    }

    // ==========================================
    // 3. EKF 更新 (雙增益融合)
    // ==========================================
    if (ndt_success) {
        Eigen::Matrix<float, 3, 1> y_res = Z_obs - H_obs * x_state;
        y_res(2) = atan2(sin(y_res(2)), cos(y_res(2))); 

        Eigen::Matrix<float, 3, 3> R_normal = R_noise_base;
        if (is_blind_exit || is_hard_turning) {
            R_normal(0,0) *= 8.0; R_normal(1,1) *= 8.0; R_normal(2,2) *= 2.0; 
        } else if (frame_id > 20 && jump_from_ekf > 1.0) {
            float penalty = exp(jump_from_ekf * 0.8);
            if (penalty > 1000.0) penalty = 1000.0;
            R_normal(0,0) *= penalty;
            R_normal(1,1) *= penalty;
        }
        Eigen::Matrix<float, 3, 3> S_normal = H_obs * P_cov * H_obs.transpose() + R_normal;
        Eigen::Matrix<float, 5, 3> K_normal = P_cov * H_obs.transpose() * S_normal.inverse();

        Eigen::Matrix<float, 3, 3> R_aggressive = R_normal;
        bool in_strong_correction_zone = false;
        
        // if (frame_id > 1200 && frame_id < 1300 && jump_from_ekf < 0.6) {
        //     R_aggressive(0,0) = 0.04;
        //     R_aggressive(1,1) = 0.04;
        //     R_aggressive(2,2) = 0.08;
        //     Q_noise(4,4) = 0.04; 
        //     in_strong_correction_zone = true;
        // }

        Eigen::Matrix<float, 3, 3> S_aggressive = H_obs * P_cov * H_obs.transpose() + R_aggressive;
        Eigen::Matrix<float, 5, 3> K_aggressive = P_cov * H_obs.transpose() * S_aggressive.inverse();

        Eigen::Matrix<float, 5, 3> K_final;
        if (in_strong_correction_zone) {
            K_final.row(0) = K_aggressive.row(0); 
            K_final.row(1) = K_aggressive.row(1); 
            K_final.row(2) = K_aggressive.row(2); 
            K_final.row(3) = K_normal.row(3);     
            K_final.row(4) = K_normal.row(4);     
        } else {
            K_final = K_normal;
        }

        if (is_hard_turning || is_blind_exit) {
            K_final.row(3).setZero(); 
        }

        K_final.row(3).setZero();
        
        float v_before_update = x_state(3); 
        x_state = x_state + K_final * y_res;
        x_state(2) = atan2(sin(x_state(2)), cos(x_state(2))); 
        
        if (x_state(3) > 15.0) x_state(3) = 15.0; 
        
        // 【已修復】加入 turn_factor 宣告
        float turn_factor = std::min(abs(x_state(4)) / 0.25f, 1.0f); 
        float max_dv_pos = (1.0 + turn_factor * 4.0) * total_dt_for_clamp; 
        float max_dv_neg = (1.5 + turn_factor * 6.0) * total_dt_for_clamp; 

        if (x_state(3) - v_before_update > max_dv_pos) {
            x_state(3) = v_before_update + max_dv_pos; 
        } else if (v_before_update - x_state(3) > max_dv_neg) {
            x_state(3) = v_before_update - max_dv_neg; 
        }

        if (abs(x_state(3)) < 0.3) x_state(3) = 0.0; 
        else if (x_state(3) < 0.0) x_state(3) = 0.0; 
        if (x_state(3) == 0.0) x_state(4) = 0.0; 

        Eigen::Matrix<float, 5, 5> I5 = Eigen::Matrix<float, 5, 5>::Identity();
        P_cov = (I5 - K_final * H_obs) * P_cov;

    } else {
        x_state(3) *= 0.98; 
        if (x_state(3) < 0.0) x_state(3) = 0.0;
    }

    transform_matrix(0,3) = x_state(0); transform_matrix(1,3) = x_state(1);
    transform_matrix(0,0) = cos(x_state(2)); transform_matrix(0,1) = -sin(x_state(2));
    transform_matrix(1,0) = sin(x_state(2)); transform_matrix(1,1) = cos(x_state(2));
    transform_matrix(2,3) = init_guess(2,3) * 0.8; 

    radar_pose_publisher(transform_matrix);

    sensor_msgs::PointCloud2 radar_msg;
    pcl::toROSMsg(*radar_pc, radar_msg);
    radar_msg.header.frame_id = "radar"; 
    radar_msg.header.stamp = ros::Time::now();
    pc_pub.publish(radar_msg);

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
    checkpoint_file = pkg + "/result/checkpoint.txt"; 

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/radar", 10);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/visual/tranformed_radar_pose", 10);
    path_pub = nh.advertise<nav_msgs::Path>("/visual/localization_path", 10);
    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/map", 1, true);

    if (LOAD_CHECKPOINT) {
        file.open(save_file, std::ios::app);
    } else {
        file.open(save_file);
    }

    string imu_file = pkg + "/data/Track2/imu_data.csv";
    ifstream imu_in(imu_file);
    if(imu_in.is_open()) {
        string imu_line;
        getline(imu_in, imu_line); 
        while(getline(imu_in, imu_line)) {
            stringstream ss(imu_line);
            string cell; vector<string> tokens;
            while(getline(ss, cell, ',')) tokens.push_back(cell);
            if(tokens.size() > 29) {
                try {
                    double t = stod(tokens[0]);
                    if (t > 1e15) t = t / 1e9; 
                    float wz = stof(tokens[19]);
                    float ax = stof(tokens[29]);
                    imu_data_map[t] = {wz, ax}; 
                } catch(...) {}
            }
        }
        imu_in.close();
    }

    ROS_INFO("Loading and Publishing Global Map...");
    pcl::io::loadPCDFile<pcl::PointXYZ>(map_file, *map_pc);
    sensor_msgs::PointCloud2 map_msg;
    pcl::toROSMsg(*map_pc, map_msg);
    map_msg.header.frame_id = "map";
    map_msg.header.stamp = ros::Time::now();
    map_pub.publish(map_msg);

    ifstream t_in(t_file); string line;
    int f_idx = 0;
    double last_radar_time = -1.0; 

    if (LOAD_CHECKPOINT) {
        ifstream cp_in(checkpoint_file);
        if (cp_in.is_open()) {
            cp_in >> f_idx >> last_radar_time;
            for(int i=0; i<5; i++) cp_in >> x_state(i);
            for(int i=0; i<5; i++) for(int j=0; j<5; j++) cp_in >> P_cov(i,j);
            for(int i=0; i<4; i++) for(int j=0; j<4; j++) cp_in >> transform_matrix(i,j);
            cp_in.close();
            
            checkpoint_loaded = true;
            ROS_INFO(">>> Checkpoint Loaded! Resuming from frame %d <<<", f_idx);

            string dummy;
            for(int i = 0; i < f_idx; i++) {
                getline(t_in, dummy);
            }
        } else {
            ROS_ERROR("Checkpoint file not found! Starting from 0.");
        }
    }
    
    ROS_INFO("Starting EKF Loop...");
    while(getline(t_in, line) && ros::ok()) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_folder + "/" + line + ".pcd", *radar_pc) != -1) {
            
            double current_radar_time = stod(line);
            if (current_radar_time > 1e15) current_radar_time /= 1e9; 

            std::vector<std::pair<double, ImuData>> imu_seq;
            if (last_radar_time > 0.0) {
                auto it_start = imu_data_map.upper_bound(last_radar_time);
                auto it_end = imu_data_map.upper_bound(current_radar_time);
                for (auto it = it_start; it != it_end; ++it) {
                    imu_seq.push_back(*it);
                }
            }
            
            callback(f_idx++, imu_seq, last_radar_time, current_radar_time);
            last_radar_time = current_radar_time;

            if (SAVE_CHECKPOINT && (f_idx - 1) == TARGET_CHECKPOINT_FRAME) {
                ofstream cp_out(checkpoint_file);
                cp_out << (f_idx - 1) << " " << std::fixed << std::setprecision(9) << last_radar_time << "\n";
                for(int i=0; i<5; i++) cp_out << x_state(i) << " "; cp_out << "\n";
                for(int i=0; i<5; i++) for(int j=0; j<5; j++) cp_out << P_cov(i,j) << " "; cp_out << "\n";
                for(int i=0; i<4; i++) for(int j=0; j<4; j++) cp_out << transform_matrix(i,j) << " "; cp_out << "\n";
                cp_out.close();
                ROS_WARN(">>> Checkpoint Saved at frame %d! <<<", TARGET_CHECKPOINT_FRAME);
            }

            ros::spinOnce();
            ros::Duration(0.01).sleep(); 
        }
    }
    
    file.close();
    ROS_INFO("Processing finished. Keep spinning for RViz.");
    ros::spin(); 
    return 0;
}