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

// ==========================================
// GTSAM 相關標頭檔
// ==========================================
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/linear/NoiseModel.h>

using gtsam::symbol_shorthand::X; // Pose3
using gtsam::symbol_shorthand::V; // Velocity3
using gtsam::symbol_shorthand::B; // imuBias::ConstantBias
using namespace std;

// ==========================================
// 資料結構定義
// ==========================================
struct PointRadar {
    float x; float y; float z;
    float range_rate; float rcs;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointRadar,
                                  (float, x, x)(float, y, y)(float, z, z)
                                  (float, range_rate, RangeRate)(float, rcs, RCS))

struct ImuData {
    float wx, wy, wz;
    float ax, ay, az;
};

// ==========================================
// 全域變數與參數
// ==========================================
ros::Publisher pc_pub, radar_pose_pub, path_pub, map_pub;
nav_msgs::Path path_msg;
ofstream file;
std::map<double, ImuData> imu_data_map;
string map_file, save_file, t_file, pcd_folder;

pcl::PointCloud<PointRadar>::Ptr raw_radar_pc (new pcl::PointCloud<PointRadar>);
pcl::PointCloud<pcl::PointXYZ>::Ptr radar_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr map_pc (new pcl::PointCloud<pcl::PointXYZ>);
pcl::PointCloud<pcl::PointXYZ>::Ptr submap_pc (new pcl::PointCloud<pcl::PointXYZ>);

// GTSAM 全域物件
std::unique_ptr<gtsam::ISAM2> isam;
gtsam::NonlinearFactorGraph graph;
gtsam::Values initial_estimates;
std::shared_ptr<gtsam::PreintegratedImuMeasurements> imu_preintegrator;

int current_key = 0;
bool initialized = false;

// NDT Keyframe 變數 (控制彎道閉眼)
bool first_ndt = true;
float last_ndt_yaw = 0.0;
float last_ndt_x = 0.0, last_ndt_y = 0.0;

Eigen::Matrix4f current_transform_matrix = Eigen::Matrix4f::Identity();

// [參數調校區] 初始位置 (Initial Guess)
Eigen::Matrix4f getInitialGuessMatrix() {
    Eigen::Matrix4f mat;
    mat <<  0.699603508,   -0.712504332,   -0.053782041,  169.901155884,
            0.713430858,    0.700720265,   -0.002742425,  -31.630335462,
            0.039640155,   -0.036451157,    0.998548932,    0.084140257,
            0.000000000,    0.000000000,    0.000000000,    1.000000000;
    return mat;
}
const float INITIAL_VELOCITY = 5.5; // 初始預設車速 (m/s)

// ==========================================
// 初始化 GTSAM 系統
// ==========================================
void initGTSAM() {
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1;
    isam = std::make_unique<gtsam::ISAM2>(parameters);
    
    auto preint_params = gtsam::PreintegrationParams::MakeSharedU(9.81); 
    preint_params->accelerometerCovariance = gtsam::Matrix33::Identity() * 0.01; 
    preint_params->gyroscopeCovariance = gtsam::Matrix33::Identity() * 0.00001; 
    preint_params->integrationCovariance = gtsam::Matrix33::Identity() * 1e-8; 

    imu_preintegrator = std::make_shared<gtsam::PreintegratedImuMeasurements>(
        preint_params, gtsam::imuBias::ConstantBias());
}

// ==========================================
// 輔助函式
// ==========================================
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

bool estimateVelocityFromRadar(const pcl::PointCloud<PointRadar>::Ptr& cloud, float& est_vx, float& est_vy) {
    std::vector<PointRadar> valid_pts;
    for (const auto& p : cloud->points) {
        float d = sqrt(p.x * p.x + p.y * p.y);
        if (d > 2.0 && std::isfinite(p.range_rate)) valid_pts.push_back(p);
    }
    if (valid_pts.size() < 10) return false;

    int max_inliers = 0;
    float best_vx = 0.0f, best_vy = 0.0f;
    float inlier_thresh = 0.4; 

    for (int i = 0; i < 100; ++i) {
        int idx1 = rand() % valid_pts.size(); int idx2 = rand() % valid_pts.size();
        if (idx1 == idx2) continue;

        PointRadar p1 = valid_pts[idx1]; PointRadar p2 = valid_pts[idx2];
        float d1 = sqrt(p1.x*p1.x + p1.y*p1.y); float d2 = sqrt(p2.x*p2.x + p2.y*p2.y);
        float u1x = p1.x/d1, u1y = p1.y/d1; float u2x = p2.x/d2, u2y = p2.y/d2;

        float det = u1x*u2y - u2x*u1y;
        if (abs(det) < 0.1) continue; 

        float vx = (u2y * (-p1.range_rate) - u1y * (-p2.range_rate)) / det;
        float vy = (-u2x * (-p1.range_rate) + u1x * (-p2.range_rate)) / det;

        int inliers = 0;
        for (const auto& p : valid_pts) {
            float d = sqrt(p.x*p.x + p.y*p.y);
            float expected_vr = -(p.x/d * vx + p.y/d * vy); 
            if (abs(p.range_rate - expected_vr) < inlier_thresh) inliers++;
        }
        if (inliers > max_inliers) { max_inliers = inliers; best_vx = vx; best_vy = vy; }
    }

    if (best_vx < -5.0 || best_vx > 40.0) return false;

    est_vx = best_vx; 
    est_vy = best_vy; 
    return true;
}

// ==========================================
// 核心回呼函式 (GTSAM 因子圖構建與最佳化)
// ==========================================
void callback(int frame_id, const std::vector<std::pair<double, ImuData>>& imu_seq, double last_radar_time, double current_radar_time) {
    if(!initialized) {
        initGTSAM();
        current_transform_matrix = getInitialGuessMatrix();
        current_key = 0;
        initialized = true;
    }

    // ----------------------------------------------------
    // 1. 第一幀：加入起始先驗約束 (Prior Factors)
    // ----------------------------------------------------
    if (current_key == 0) {
        gtsam::Pose3 init_pose(current_transform_matrix.cast<double>());
        double init_yaw = atan2(current_transform_matrix(1, 0), current_transform_matrix(0, 0));
        gtsam::Vector3 init_vel(INITIAL_VELOCITY * cos(init_yaw), INITIAL_VELOCITY * sin(init_yaw), 0.0);
        gtsam::imuBias::ConstantBias init_bias;

        // 【維持你原本嚴格的起始約束】，確保開頭絕對不會因為 NDT 雜訊而被拉歪！
        gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, M_PI, 1e-1, 1e-1, 1e-1).finished());
        gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise = gtsam::noiseModel::Isotropic::Sigma(3, 1e-1);
        gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3);

        graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(0), init_pose, priorPoseNoise));
        graph.add(gtsam::PriorFactor<gtsam::Velocity3>(V(0), init_vel, priorVelNoise));
        graph.add(gtsam::PriorFactor<gtsam::imuBias::ConstantBias>(B(0), init_bias, priorBiasNoise));

        initial_estimates.insert(X(0), init_pose);
        initial_estimates.insert(V(0), init_vel);
        initial_estimates.insert(B(0), init_bias);

        isam->update(graph, initial_estimates);
        isam->update();
        graph.resize(0); initial_estimates.clear();
        
        // 【已修改：移除第一幀的即時寫檔，交給最後 Batch 處理】
        
        current_key++;
        return;
    }

    int prev_key = current_key - 1;

    radar_pc->clear();
    for (const auto& p : raw_radar_pc->points) {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z)) {
            radar_pc->push_back(pcl::PointXYZ(p.x, p.y, p.z));
        }
    }

    // ----------------------------------------------------
    // 2. IMU 預積分 (IMU Preintegration)
    // ----------------------------------------------------
    double prev_time = last_radar_time;
    for (size_t i = 0; i < imu_seq.size(); ++i) {
        double current_t = imu_seq[i].first;
        float dt = current_t - prev_time;
        if (dt <= 0.001) continue; 
        if (dt > 0.1) dt = 0.01;   

        gtsam::Vector3 measuredAcc(imu_seq[i].second.ax, imu_seq[i].second.ay, imu_seq[i].second.az);
        gtsam::Vector3 measuredOmega(imu_seq[i].second.wx, imu_seq[i].second.wy, imu_seq[i].second.wz);
        imu_preintegrator->integrateMeasurement(measuredAcc, measuredOmega, dt);
        prev_time = current_t;
    }

    float dt_remainder = current_radar_time - prev_time;
    if (dt_remainder > 0.001 && dt_remainder < 0.1 && !imu_seq.empty()) {
        gtsam::Vector3 lastAcc(imu_seq.back().second.ax, imu_seq.back().second.ay, imu_seq.back().second.az);
        gtsam::Vector3 lastOmega(imu_seq.back().second.wx, imu_seq.back().second.wy, imu_seq.back().second.wz);
        imu_preintegrator->integrateMeasurement(lastAcc, lastOmega, dt_remainder);
    }

    graph.add(gtsam::ImuFactor(X(prev_key), V(prev_key), X(current_key), V(current_key), B(prev_key), *imu_preintegrator));

    gtsam::noiseModel::Diagonal::shared_ptr bias_rw_noise = gtsam::noiseModel::Isotropic::Sigma(6, 1e-4);
    graph.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(prev_key), B(current_key), gtsam::imuBias::ConstantBias(), bias_rw_noise));

    gtsam::NavState prev_state(isam->calculateEstimate<gtsam::Pose3>(X(prev_key)), isam->calculateEstimate<gtsam::Velocity3>(V(prev_key)));
    gtsam::NavState prop_state = imu_preintegrator->predict(prev_state, isam->calculateEstimate<gtsam::imuBias::ConstantBias>(B(prev_key)));
    
    initial_estimates.insert(X(current_key), prop_state.pose());
    initial_estimates.insert(V(current_key), prop_state.velocity());
    initial_estimates.insert(B(current_key), isam->calculateEstimate<gtsam::imuBias::ConstantBias>(B(prev_key)));

    // ----------------------------------------------------
    // 3. 雷達都卜勒速度因子 + 精確外參剛體運動學 + ZUPT
    // ----------------------------------------------------
    float obs_vx = 0.0, obs_vy = 0.0;
    bool has_vel = estimateVelocityFromRadar(raw_radar_pc, obs_vx, obs_vy);
    bool is_stopped = has_vel && (abs(obs_vx) < 0.1);

    if (is_stopped) {
        gtsam::noiseModel::Diagonal::shared_ptr zuptNoise = gtsam::noiseModel::Diagonal::Sigmas(gtsam::Vector3(1e-4, 1e-4, 1e-4));
        graph.add(gtsam::PriorFactor<gtsam::Velocity3>(V(current_key), gtsam::Vector3(0, 0, 0), zuptNoise));
        
        gtsam::noiseModel::Diagonal::shared_ptr poseLockNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5).finished());
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(X(prev_key), X(current_key), gtsam::Pose3(), poseLockNoise));
    } 
    else if (has_vel) {
        gtsam::Rot3 pred_rot = prop_state.pose().rotation();
        float current_wz = 0.0;
        if (!imu_seq.empty()) current_wz = imu_seq.back().second.wz;

        // ==================================================
        // 【核心突破：精確外參剛體運動學】
        // 已知 IMU 位於 X = -2.0 (與後軸重合)
        // 已知 Radar 位於 X = 2.022, Y = 0.359
        // Radar 到 IMU 的 Y 軸臂長 (Lever Arm Y) = 0.359
        // ==================================================
        const float LEVER_ARM_Y = 0.359; 
        
        // 運動學反推：雷達的測量值，加上旋轉補償，得到真正的後軸(IMU)前進速度
        float v_imu_x = obs_vx + current_wz * LEVER_ARM_Y;
        
        // 因為 IMU 完美地位於車輛後輪軸，物理上不允許側滑！
        float v_imu_y = 0.0; 
        
        gtsam::Vector3 local_vel(v_imu_x, v_imu_y, 0.0); 
        gtsam::Vector3 world_vel_obs = pred_rot.rotate(local_vel);
        
        // ==================================================
        // 【車體座標系協方差矩陣】
        // 因為我們已經把速度推算到了完美的「後軸中心」
        // 我們再也不需要根據轉彎來切換雜訊了！
        // ==================================================
        double sigma_x = 0.2;  // 信任雷達測出的前進速度
        double sigma_y = 0.02; // ★死死鎖住後軸，不准螃蟹走★
        double sigma_z = 1e-4; // 鎖死垂直速度
        
        Eigen::Matrix3d body_cov = Eigen::Vector3d(sigma_x*sigma_x, sigma_y*sigma_y, sigma_z*sigma_z).asDiagonal();
        Eigen::Matrix3d R = pred_rot.matrix();
        Eigen::Matrix3d world_cov = R * body_cov * R.transpose(); // 旋轉到世界座標系
        
        gtsam::noiseModel::Gaussian::shared_ptr radarVelNoise = gtsam::noiseModel::Gaussian::Covariance(world_cov);
        graph.add(gtsam::PriorFactor<gtsam::Velocity3>(V(current_key), world_vel_obs, radarVelNoise));
    }
    // ----------------------------------------------------
    // 3.5 偽 2D 約束 (防止 Z 軸高度飄移影響 ATE)
    // ----------------------------------------------------
    if (!is_stopped) {
        double initial_z = current_transform_matrix(2, 3); // 鎖定在初始高度
        gtsam::Pose3 flat_pose(gtsam::Rot3::Yaw(prop_state.pose().rotation().yaw()), 
                               gtsam::Point3(prop_state.pose().translation().x(), prop_state.pose().translation().y(), initial_z));
        
        gtsam::noiseModel::Diagonal::shared_ptr pseudo2dNoise = gtsam::noiseModel::Diagonal::Sigmas(
            (gtsam::Vector(6) << 1e-3, 1e-3, 1000.0, 1000.0, 1000.0, 1e-3).finished()); 
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(current_key), flat_pose, pseudo2dNoise));
    }

    // ----------------------------------------------------
    // 4. 地圖匹配 (NDT Factor) - 彎道閉眼
    // ----------------------------------------------------
    float current_yaw = prop_state.pose().rotation().yaw();
    float current_x = prop_state.pose().translation().x();
    float current_y = prop_state.pose().translation().y();
    float dist_diff = sqrt(pow(current_x - last_ndt_x, 2) + pow(current_y - last_ndt_y, 2));
    float yaw_diff = abs(current_yaw - last_ndt_yaw);

    // 每移動 0.5m 且不在轉彎時才做 NDT (直道更頻繁對齊，防止漂移)
    // bool should_do_ndt = first_ndt || ((dist_diff > 0.2) && (yaw_diff < 0.03));
    bool should_do_ndt = true;

    if (!is_stopped && should_do_ndt) {
        Eigen::Matrix4f init_guess = prop_state.pose().matrix().cast<float>();
    
        submap_pc->clear();
        pcl::PointCloud<pcl::PointXYZ>::Ptr coarse_submap (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::CropBox<pcl::PointXYZ> lidar_crop;
        lidar_crop.setInputCloud(map_pc);
        lidar_crop.setMin(Eigen::Vector4f(0.0, -45.0, -100.0, 1.0)); 
        lidar_crop.setMax(Eigen::Vector4f(120.0, 45.0,  100.0, 1.0)); 
        lidar_crop.setTranslation(Eigen::Vector3f(init_guess(0,3), init_guess(1,3), 0.0));
        
        float pred_yaw = atan2(init_guess(1,0), init_guess(0,0));
        lidar_crop.setRotation(Eigen::Vector3f(0.0, 0.0, pred_yaw));
        lidar_crop.filter(*coarse_submap);

        float ego_x = init_guess(0,3); float ego_y = init_guess(1,3);
        float cos_yaw = cos(-pred_yaw); float sin_yaw = sin(-pred_yaw);

        for (const auto& p : coarse_submap->points) {
            float dx_global = p.x - ego_x; float dy_global = p.y - ego_y;
            float dx_local = dx_global * cos_yaw - dy_global * sin_yaw;
            float dy_local = dx_global * sin_yaw + dy_global * cos_yaw;

            if (dx_local < -20.0) continue;
            
            float distance = sqrt(dx_local * dx_local + dy_local * dy_local);
            float angle_deg = atan2(dy_local, dx_local) * 180.0 / M_PI;

            bool in_low_beam = (distance <= 45.0 && abs(angle_deg) <= 60.0);
            bool in_high_beam = (distance <= 120.0 && abs(angle_deg) <= 20.0);
            if (in_low_beam || in_high_beam) submap_pc->push_back(p);
        }

        if (!submap_pc->empty() && radar_pc->size() > 10) {
            pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
            static pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
            ndt.setInputSource(radar_pc);
            ndt.setInputTarget(submap_pc);
            ndt.setTransformationEpsilon(1e-5);

            ndt.setResolution(5.0); ndt.setStepSize(0.2); ndt.setMaximumIterations(80);
            ndt.align(*tmp, init_guess);
            Eigen::Matrix4f res_mid = ndt.getFinalTransformation();
            
            ndt.setResolution(1.5); 
            ndt.setStepSize(0.05); 
            ndt.setMaximumIterations(100);
            ndt.align(*tmp, res_mid);

            double fitness_score = ndt.getFitnessScore(1.0);
            if (ndt.hasConverged() && fitness_score < 0.6) {
                gtsam::Pose3 ndt_pose(ndt.getFinalTransformation().cast<double>());
                double ndt_sigma = 0.05 + (fitness_score * fitness_score * 2.0); 

                // 【沒有 Huber】維持你原本的邏輯，不干擾 GTSAM 的信心
                gtsam::noiseModel::Diagonal::shared_ptr ndtNoise = gtsam::noiseModel::Diagonal::Sigmas(
                    (gtsam::Vector(6) << 0.05, 0.05, 0.1, ndt_sigma, ndt_sigma, ndt_sigma * 1.5).finished()); 
                
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(X(current_key), ndt_pose, ndtNoise));
                
                // 更新 NDT 關鍵幀狀態
                first_ndt = false;
                last_ndt_x = current_x; 
                last_ndt_y = current_y; 
                last_ndt_yaw = current_yaw;
            }
            // 條件稍微收緊，太差的配準直接丟棄
            
        }
    }

    // ----------------------------------------------------
    // 5. 執行 ISAM2 最佳化與更新狀態
    // ----------------------------------------------------
    isam->update(graph, initial_estimates);
    isam->update(); // 執行兩次確保收斂
    
    graph.resize(0);
    initial_estimates.clear();

    gtsam::Pose3 current_pose_gtsam = isam->calculateEstimate<gtsam::Pose3>(X(current_key));
    gtsam::imuBias::ConstantBias current_bias = isam->calculateEstimate<gtsam::imuBias::ConstantBias>(B(current_key));
    
    imu_preintegrator->resetIntegrationAndSetBias(current_bias);

    // 保留即時發布給 RViz 顯示 (但不再寫檔)
    current_transform_matrix = current_pose_gtsam.matrix().cast<float>();
    radar_pose_publisher(current_transform_matrix);

    sensor_msgs::PointCloud2 radar_msg;
    pcl::toROSMsg(*radar_pc, radar_msg);
    radar_msg.header.frame_id = "radar"; 
    radar_msg.header.stamp = ros::Time::now();
    pc_pub.publish(radar_msg);

    // 【已修改：移除中途的 file 寫檔迴圈】

    current_key++;
}

// ==========================================
// Main 函式
// ==========================================
int main(int argc, char** argv) {
    ros::init(argc, argv, "testa_node");
    ros::NodeHandle nh;
    string pkg = ros::package::getPath("localization");
    t_file = pkg + "/data/Track2/radar_t_list.txt";
    pcd_folder = pkg + "/data/Track2/raw_radar_pcd";
    map_file = pkg + "/data/Track2/map_modified.pcd";
    save_file = pkg + "/result/result.txt";

    pc_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/radar", 10);
    radar_pose_pub = nh.advertise<geometry_msgs::PoseStamped>("/visual/tranformed_radar_pose", 10);
    path_pub = nh.advertise<nav_msgs::Path>("/visual/localization_path", 10);
    map_pub = nh.advertise<sensor_msgs::PointCloud2>("/visual/map", 1, true);

    file.open(save_file);
    
    double IMU_TIME_OFFSET = 0.00; 
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
                    float wx = stof(tokens[17]); float wy = stof(tokens[18]); float wz = stof(tokens[19]);
                    float ax = stof(tokens[29]); float ay = stof(tokens[30]); float az = stof(tokens[31]);
                    imu_data_map[t] = {wx, wy, wz, ax, ay, az}; 
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

    ROS_INFO("Starting GTSAM ISAM2 Loop...");
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

            ros::spinOnce();
            ros::Duration(0.01).sleep(); 
        }
    }
    
    // =========================================================
    // 【新增核心功能】在所有資料跑完後，提取並寫入最終的平滑軌跡
    // =========================================================
    ROS_INFO("Data playback finished. Extracting smoothed trajectory from GTSAM...");
    
    if (isam) {
        gtsam::Values final_results = isam->calculateEstimate();
        
        // 從 Frame 0 一路印到最後一個 Frame
        for (int i = 0; i < current_key; ++i) {
            if (final_results.exists(X(i))) {
                gtsam::Pose3 optimal_pose = final_results.at<gtsam::Pose3>(X(i));
                Eigen::Matrix4f mat = optimal_pose.matrix().cast<float>();
                
                // 依照原始格式寫入檔案
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 4; ++c) {
                        file << mat(r, c) << ( (r==2 && c==3) ? "" : " " );
                    }
                }
                file << endl;
            }
        }
    }
    ROS_INFO("Trajectory saved successfully! Total frames: %d", current_key);
    // =========================================================

    file.close();
    ROS_INFO("Processing finished. Keep spinning for RViz.");
    ros::spin(); 
    return 0;
}