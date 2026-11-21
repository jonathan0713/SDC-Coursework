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
#include <pcl/registration/icp.h> // <--- 【新增這一行】
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
// 自訂 Radar 點雲結構 (解析 RangeRate 與 RCS)
// ==========================================
struct PointRadar {
    float x;
    float y;
    float z;
    float range_rate;
    float rcs;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointRadar,
                                  (float, x, x)
                                  (float, y, y)
                                  (float, z, z)
                                  (float, range_rate, RangeRate)
                                  (float, rcs, RCS))

// ==========================================
// Checkpoint 控制面板
// ==========================================
bool LOAD_CHECKPOINT = false;         
bool SAVE_CHECKPOINT = true;          
int TARGET_CHECKPOINT_FRAME = 1000;   
bool checkpoint_loaded = false;       

// ==========================================
// [參數調校區] 初始位置 (Initial Guess)
// ==========================================
Eigen::Matrix4f getInitialGuessMatrix() {
    Eigen::Matrix4f mat;
    mat <<  0.699603508,   -0.712504332,   -0.053782041,  169.901155884,
            0.713430858,    0.700720265,   -0.002742425,  -31.630335462,
            0.039640155,   -0.036451157,    0.998548932,    0.084140257,
            0.000000000,    0.000000000,    0.000000000,    1.000000000;
    return mat;
}
const float INITIAL_VELOCITY = 5.5; // EKF 初始預設車速 (m/s)

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

// 儲存 IMU 數據的結構體
struct ImuData {
    float wz;
    float ax;
};
std::map<double, ImuData> imu_data_map;

// EKF 狀態矩陣宣告
Eigen::Matrix<float, 5, 1> x_state;
Eigen::Matrix<float, 5, 5> P_cov;
Eigen::Matrix<float, 5, 5> Q_noise;
Eigen::Matrix<float, 3, 5> H_obs;
Eigen::Matrix<float, 3, 3> R_noise_base; 

// 點雲指標宣告
pcl::PointCloud<PointRadar>::Ptr raw_radar_pc (new pcl::PointCloud<PointRadar>);
pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);

string map_file, save_file, t_file, pcd_folder, checkpoint_file; 

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

// ==========================================
// Radar 都卜勒速度估測器 (RANSAC + Least Squares)
// ==========================================
bool estimateVelocityFromRadar(const pcl::PointCloud<PointRadar>::Ptr& cloud, float& estimated_v) {
    std::vector<PointRadar> valid_pts;
    for (const auto& p : cloud->points) {
        float d = sqrt(p.x * p.x + p.y * p.y);
        if (d > 2.0 && std::isfinite(p.range_rate)) {
            valid_pts.push_back(p);
        }
    }
    
    if (valid_pts.size() < 10) return false;

    int max_inliers = 0;
    float best_vx = 0.0f, best_vy = 0.0f;
    float inlier_thresh = 0.4; 

    for (int i = 0; i < 100; ++i) {
        int idx1 = rand() % valid_pts.size();
        int idx2 = rand() % valid_pts.size();
        if (idx1 == idx2) continue;

        PointRadar p1 = valid_pts[idx1];
        PointRadar p2 = valid_pts[idx2];

        float d1 = sqrt(p1.x*p1.x + p1.y*p1.y);
        float d2 = sqrt(p2.x*p2.x + p2.y*p2.y);

        float u1x = p1.x/d1, u1y = p1.y/d1;
        float u2x = p2.x/d2, u2y = p2.y/d2;

        float det = u1x*u2y - u2x*u1y;
        if (abs(det) < 0.1) continue; 

        float vx = (u2y * (-p1.range_rate) - u1y * (-p2.range_rate)) / det;
        float vy = (-u2x * (-p1.range_rate) + u1x * (-p2.range_rate)) / det;

        int inliers = 0;
        for (const auto& p : valid_pts) {
            float d = sqrt(p.x*p.x + p.y*p.y);
            float ux = p.x/d, uy = p.y/d;
            float expected_vr = -(ux * vx + uy * vy); 
            if (abs(p.range_rate - expected_vr) < inlier_thresh) {
                inliers++;
            }
        }

        if (inliers > max_inliers) {
            max_inliers = inliers;
            best_vx = vx;
            best_vy = vy;
        }
    }

    if (max_inliers < 15) return false; 

    float sum_ux2 = 0, sum_uy2 = 0, sum_uxuy = 0;
    float sum_uxvr = 0, sum_uyvr = 0;
    
    for (const auto& p : valid_pts) {
        float d = sqrt(p.x*p.x + p.y*p.y);
        float ux = p.x/d, uy = p.y/d;
        float expected_vr = -(ux * best_vx + uy * best_vy);
        
        if (abs(p.range_rate - expected_vr) < inlier_thresh) {
            sum_ux2 += ux*ux;
            sum_uy2 += uy*uy;
            sum_uxuy += ux*uy;
            sum_uxvr += ux * (-p.range_rate);
            sum_uyvr += uy * (-p.range_rate);
        }
    }
    
    float det = sum_ux2 * sum_uy2 - sum_uxuy * sum_uxuy;
    if (abs(det) > 0.001) {
        best_vx = (sum_uy2 * sum_uxvr - sum_uxuy * sum_uyvr) / det;
        best_vy = (sum_ux2 * sum_uyvr - sum_uxuy * sum_uxvr) / det;
    }

    if (best_vx < -5.0 || best_vx > 40.0) return false;

    estimated_v = best_vx; 
    return true;
}

void callback(int frame_id, const std::vector<std::pair<double, ImuData>>& imu_seq, double last_radar_time, double current_radar_time) {
    if(!initialized) {
        if (!checkpoint_loaded) {
            transform_matrix = getInitialGuessMatrix();

            x_state.setZero();
            x_state(0) = transform_matrix(0, 3); 
            x_state(1) = transform_matrix(1, 3); 
            x_state(2) = atan2(transform_matrix(1, 0), transform_matrix(0, 0)); 
            
            x_state(3) = INITIAL_VELOCITY; 
            x_state(4) = 0.0; 

            P_cov.setZero(); 
            P_cov(0,0) = 1.0; P_cov(1,1) = 1.0; P_cov(2,2) = 0.1; P_cov(3,3) = 0.6; P_cov(4,4) = 0.6; 
        }
        
        Q_noise.setIdentity();
        Q_noise(0,0) = 0.001;  
        Q_noise(1,1) = 0.001;  
        Q_noise(2,2) = 0.0025; 
        Q_noise(3,3) = 0.2;    
        Q_noise(4,4) = 0.001; 

        H_obs.setZero(); 
        H_obs(0,0) = 1.0; H_obs(1,1) = 1.0; H_obs(2,2) = 1.0; 

        R_noise_base.setIdentity(); 
        R_noise_base(0,0) = 4; R_noise_base(1,1) = 4; R_noise_base(2,2) = 0.3; 
        initialized = true;
    }

    radar_pc->clear();
    for (const auto& p : raw_radar_pc->points) {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            radar_pc->push_back(pcl::PointXYZ(p.x, p.y, p.z));
        }
    }

    // ==========================================
    // 1. EKF 預測
    // ==========================================
    const float IMU_HARDWARE_DELAY = 0.0f; 
    const float ALPHA_LPF_GAIN = 1.0f;      

    static float prev_raw_wz = 0.0f;
    static double prev_imu_time = 0.0;
    static float filtered_angular_accel = 0.0f;
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
            
            // ==========================================
            // 【新增 Debug 輸出】印出實際讀取到的 IMU Yaw Rate
            // ==========================================
            // 印出所有數值供觀察。如果覺得洗版太快，可以包在 if(frame_id % 5 == 0) 裡面
            ROS_INFO("FRM %d | IMU Raw wz: %.5f", frame_id, raw_wz);
            // ==========================================
            
            // 【新增】 IMU Yaw Rate Deadband (死區過濾)
            const float IMU_YAW_RATE_DEADBAND = 0.02f; 
            if (std::abs(raw_wz) < IMU_YAW_RATE_DEADBAND) {
                raw_wz = 0.0f;
            }
            // ==========================================
            
            float dt = current_t - prev_time;
            if (dt <= 0.001) continue; 
            if (dt > 0.1) dt = 0.01;   

            float imu_dt = current_t - prev_imu_time;
            if (imu_dt > 0.0 && imu_dt < 0.1) {
                float raw_accel = (raw_wz - prev_raw_wz) / imu_dt;
                filtered_angular_accel = (1.0f - ALPHA_LPF_GAIN) * filtered_angular_accel + ALPHA_LPF_GAIN * raw_accel;
            }
            
            float compensated_wz = raw_wz + IMU_HARDWARE_DELAY * filtered_angular_accel;
            x_state(4) = compensated_wz; 

            prev_raw_wz = raw_wz;
            prev_imu_time = current_t;
            
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
            float exact_wz_at_radar_time = x_state(4); 
            float dt = dt_remainder;
            float x = x_state(0), y = x_state(1), yaw = x_state(2), v = x_state(3), w = exact_wz_at_radar_time;
            
            x_state(0) = x + v * dt * cos(yaw); 
            x_state(1) = y + v * dt * sin(yaw); 
            x_state(2) = yaw + w * dt;
            x_state(2) = atan2(sin(x_state(2)), cos(x_state(2)));

            Eigen::Matrix<float, 5, 5> F_jacob = Eigen::Matrix<float, 5, 5>::Identity();
            F_jacob(0, 2) = -v * dt * sin(yaw); F_jacob(0, 3) = cos(yaw) * dt;
            F_jacob(1, 2) = v * dt * cos(yaw);  F_jacob(1, 3) = sin(yaw) * dt; F_jacob(2, 4) = dt;

            P_cov = F_jacob * P_cov * F_jacob.transpose() + Q_noise * (dt / 0.05f);
        }
        total_dt_for_clamp = current_radar_time - last_radar_time;
    }

    // ==========================================
    // 1.5. 雷達 Doppler 獨立速度更新
    // ==========================================
    float obs_v = 0.0;
    if (estimateVelocityFromRadar(raw_radar_pc, obs_v)) {
        Eigen::Matrix<float, 1, 5> H_vel;
        H_vel << 0, 0, 0, 1, 0;

        float y_vel = obs_v - x_state(3); 
        float R_vel = 0.2; 

        float S_vel = H_vel * P_cov * H_vel.transpose() + R_vel;
        Eigen::Matrix<float, 5, 1> K_vel = P_cov * H_vel.transpose() / S_vel;

        x_state = x_state + K_vel * y_vel;
        
        Eigen::Matrix<float, 5, 5> I5 = Eigen::Matrix<float, 5, 5>::Identity();
        P_cov = (I5 - K_vel * H_vel) * P_cov;

        if (x_state(3) < 0.0) x_state(3) = 0.0;
    }

    Eigen::Matrix4f init_guess = transform_matrix;
    init_guess(0,3) = x_state(0); init_guess(1,3) = x_state(1);
    init_guess(0,0) = cos(x_state(2)); init_guess(0,1) = -sin(x_state(2));
    init_guess(1,0) = sin(x_state(2)); init_guess(1,1) = cos(x_state(2));

    // ==========================================
    // 2. 雷達與 NDT (雙扇形 Submap 擷取: 近光燈 + 遠光燈)
    // ==========================================
    // [Radar 專用] 長方形 Submap 擷取
    // ==========================================
    submap_pc->clear();
    pcl::CropBox<pcl::PointXYZ> radar_crop;
    radar_crop.setInputCloud(map_pc);

    // 雷達長方形視野 (前 120, 後 20, 左右 60)
    float radar_front = 120.0;
    float radar_back  = -20.0;
    float radar_left  = 60.0;
    float radar_right = -60.0;

    radar_crop.setMin(Eigen::Vector4f(radar_back, radar_right, -100.0, 1.0)); 
    radar_crop.setMax(Eigen::Vector4f(radar_front, radar_left,  100.0, 1.0)); 

    radar_crop.setTranslation(Eigen::Vector3f(init_guess(0,3), init_guess(1,3), 0.0));
    radar_crop.setRotation(Eigen::Vector3f(0.0, 0.0, x_state(2)));

    radar_crop.filter(*submap_pc);
    // [LiDAR 專用] 雙扇形 Submap 擷取 (近光燈 + 遠光燈)
    // ==========================================
    submap_pc->clear();

    // --- 步驟 2.1: 粗篩選 (利用 Bounding Box 快速降採樣) ---
    pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_submap (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::CropBox<pcl::PointXYZ> lidar_crop;
    lidar_crop.setInputCloud(map_pc);

    // 粗篩選長方形只需涵蓋前方：前 120m, 左右各 45m (不抓車後)
    lidar_crop.setMin(Eigen::Vector4f(0.0, -45.0, -100.0, 1.0)); 
    lidar_crop.setMax(Eigen::Vector4f(120.0, 45.0,  100.0, 1.0)); 

    lidar_crop.setTranslation(Eigen::Vector3f(init_guess(0,3), init_guess(1,3), 0.0));
    lidar_crop.setRotation(Eigen::Vector3f(0.0, 0.0, x_state(2)));
    lidar_crop.filter(*coarse_submap);

    // --- 步驟 2.2: 精細篩選 (雙扇形邏輯) ---
    float ego_x = init_guess(0,3);
    float ego_y = init_guess(1,3);
    float ego_yaw = x_state(2);
    float cos_yaw = cos(-ego_yaw); 
    float sin_yaw = sin(-ego_yaw);

    for (const auto& p : coarse_submap->points) {
        float dx_global = p.x - ego_x;
        float dy_global = p.y - ego_y;
        
        // 轉回車體相對座標
        float dx_local = dx_global * cos_yaw - dy_global * sin_yaw;
        float dy_local = dx_global * sin_yaw + dy_global * cos_yaw;

        // 【效能優化】光達不看車後，X 為負的點直接跳過不計算三角函數
        if (dx_local < 0.0) continue;

        float distance = sqrt(dx_local * dx_local + dy_local * dy_local);
        float angle_deg = atan2(dy_local, dx_local) * 180.0 / M_PI;

        // [近光燈]：距離 45m 內，視角較廣 (左右各 60 度)
        bool in_low_beam = (distance <= 45.0 && abs(angle_deg) <= 60.0);
        
        // [遠光燈]：距離 120m 內，視角較窄 (左右各 20 度)
        bool in_high_beam = (distance <= 120.0 && abs(angle_deg) <= 20.0);
        
        if (in_low_beam || in_high_beam) {
            submap_pc->push_back(p);
        }
    }
    // std::vector<int> indices;
    // pcl::removeNaNFromPointCloud(*radar_pc, *radar_pc, indices);

    // if(frame_id>1000){
    //     pcl::PassThrough<pcl::PointXYZ> pass;
    //     pass.setInputCloud(radar_pc);
    //     pass.setFilterFieldName("z"); pass.setFilterLimits(-3.0, 10.0); pass.filter(*radar_pc);
    //     pass.setFilterFieldName("x"); pass.setFilterLimits(-80.0, 80.0); pass.filter(*radar_pc);
    //     pass.setFilterFieldName("y"); pass.setFilterLimits(-40.0, 40.0); pass.filter(*radar_pc);
    // } else {
    //     pcl::PassThrough<pcl::PointXYZ> pass;
    //     pass.setInputCloud(radar_pc);
    //     pass.setFilterFieldName("z"); pass.setFilterLimits(-3.0, 10.0); pass.filter(*radar_pc);
    //     pass.setFilterFieldName("x"); pass.setFilterLimits(-10.0, 80.0); pass.filter(*radar_pc);
    //     pass.setFilterFieldName("y"); pass.setFilterLimits(-30.0, 30.0); pass.filter(*radar_pc);
    // }
    
    // if (frame_id < 1105) {
    //     pcl::CropBox<pcl::PointXYZ> crop;
    //     crop.setInputCloud(map_pc);

    //     float front_m = 100.0;
    //     float back_m  = -100.0;
    //     float left_m  = 60.0;
    //     float right_m = -60.0;

    //     crop.setMin(Eigen::Vector4f(back_m, right_m, -100.0, 1.0)); 
    //     crop.setMax(Eigen::Vector4f(front_m, left_m,  100.0, 1.0)); 

    //     crop.setTranslation(Eigen::Vector3f(init_guess(0,3), init_guess(1,3), 0.0));
    //     crop.setRotation(Eigen::Vector3f(0.0, 0.0, x_state(2)));

    //     crop.filter(*submap_pc);
    // } 
    // else {
    //     pcl::CropBox<pcl::PointXYZ> crop;
    //     crop.setInputCloud(map_pc);

    //     float front_m = 120.0;
    //     float back_m  = -20.0;
    //     float left_m  = 60.0;
    //     float right_m = -60.0;

    //     crop.setMin(Eigen::Vector4f(back_m, right_m, -100.0, 1.0)); 
    //     crop.setMax(Eigen::Vector4f(front_m, left_m,  100.0, 1.0)); 

    //     crop.setTranslation(Eigen::Vector3f(init_guess(0,3), init_guess(1,3), 0.0));
    //     crop.setRotation(Eigen::Vector3f(0.0, 0.0, x_state(2)));

    //     crop.filter(*submap_pc);
    // }

    bool ndt_success = false;
    Eigen::Matrix<float, 3, 1> Z_obs; 
    float jump_from_ekf = 0.0;

    if (!submap_pc->empty() && radar_pc->size() > 10) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
        static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
        ndt.setInputSource(radar_pc);
        ndt.setInputTarget(submap_pc);
        ndt.setTransformationEpsilon(1e-5);

        ndt.setResolution(5.0); ndt.setStepSize(0.2); ndt.setMaximumIterations(80);
        ndt.align(*tmp, init_guess);
        Eigen::Matrix4f res_mid = ndt.getFinalTransformation();
        
        ndt.setResolution(0.3); ndt.setStepSize(0.006); ndt.setMaximumIterations(1000);
        ndt.align(*tmp, res_mid);

        double fitness_score = ndt.getFitnessScore(1.0);

        if (ndt.hasConverged() && fitness_score < 0.8) {
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
        ROS_INFO("FRM %d | v: %.2f | w: %.2f | NDT Jump: %.2fm", frame_id, x_state(3), x_state(4), jump_from_ekf);
    }

    // ==========================================
    // 3. EKF 更新
    // ==========================================
    if (ndt_success) {
        Eigen::Matrix<float, 3, 1> y_res = Z_obs - H_obs * x_state;
        y_res(2) = atan2(sin(y_res(2)), cos(y_res(2))); 

        Eigen::Matrix<float, 3, 3> R_normal = R_noise_base;
        if (frame_id > 20 && jump_from_ekf > 1.0) {
            float penalty = exp(jump_from_ekf * 0.8);
            if (penalty > 1000.0) penalty = 1000.0;
            R_normal(0,0) *= penalty;
            R_normal(1,1) *= penalty;
        }
        Eigen::Matrix<float, 3, 3> S_normal = H_obs * P_cov * H_obs.transpose() + R_normal;
        Eigen::Matrix<float, 5, 3> K_normal = P_cov * H_obs.transpose() * S_normal.inverse();

        Eigen::Matrix<float, 3, 3> R_aggressive = R_normal;
        bool in_strong_correction_zone = false;

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

        float v_before_update = x_state(3); 
        x_state = x_state + K_final * y_res;
        x_state(2) = atan2(sin(x_state(2)), cos(x_state(2))); 
        
        if (x_state(3) > 15.0) x_state(3) = 15.0; 
        
        float wz_mag = abs(x_state(4)); 
        float turn_factor = std::min(wz_mag / 0.25f, 1.0f); 

        float base_max_accel = 1.0; 
        float base_max_decel = 1.5; 
        float dynamic_max_accel = base_max_accel + (turn_factor * 4.0); 
        float dynamic_max_decel = base_max_decel + (turn_factor * 6.0);

        float max_dv_pos = dynamic_max_accel * total_dt_for_clamp; 
        float max_dv_neg = dynamic_max_decel * total_dt_for_clamp; 

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
        x_state(3) *= 1.00; 
        if (abs(x_state(3)) < 0.3) x_state(3) = 0.0; 
        else if (x_state(3) < 0.0) x_state(3) = 0.0; 
        if (x_state(3) == 0.0) x_state(4) = 0.0; 
    }

    if (x_state(3) < 0.3) x_state(3) = 0.0;

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
    
    double IMU_TIME_OFFSET = 0; 

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
                    
                    t = t + IMU_TIME_OFFSET; 
                    
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
        if (pcl::io::loadPCDFile<PointRadar>(pcd_folder + "/" + line + ".pcd", *raw_radar_pc) != -1) {
            
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