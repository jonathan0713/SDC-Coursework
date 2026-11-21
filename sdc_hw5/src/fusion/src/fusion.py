#!/usr/bin/env python3

import rospy
import os
import numpy as np
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from geometry_msgs.msg import PoseWithCovarianceStamped as Pose
from nav_msgs.msg import Odometry
from math import cos, sin, atan2, sqrt
from matplotlib import pyplot as plt

from EKF import ExtendedKalmanFilter

class Fusion:
    def __init__(self):
        rospy.Subscriber('/gps', Pose, self.gpsCallback)
        rospy.Subscriber('/radar_odometry', Odometry, self.odometryCallback)
        rospy.Subscriber('/gt_odom', Odometry, self.gtCallback)
        rospy.on_shutdown(self.shutdown)
        self.posePub = rospy.Publisher('/pred', Odometry, queue_size = 10)
        self.EKF = None
        
        self.gt_list = [[], []]
        self.est_list = [[], []]
        self.initial = False

    def shutdown(self):
        print("shutting down fusion.py")

    def predictPublish(self):
        predPose = Odometry()
        predPose.header.frame_id = 'origin'
        
        # Change to the state x and state y from EKF
        predPose.pose.pose.position.x = self.EKF.pose[0]
        predPose.pose.pose.position.y = self.EKF.pose[1]
        
        # Change to the state yaw from EKF
        quaternion = quaternion_from_euler(0, 0, self.EKF.pose[2])
        predPose.pose.pose.orientation.x = quaternion[0]
        predPose.pose.pose.orientation.y = quaternion[1]
        predPose.pose.pose.orientation.z = quaternion[2]
        predPose.pose.pose.orientation.w = quaternion[3]
        
        # Change to the covariance matrix of [x, y, yaw] from EKF
        predPose.pose.covariance = [self.EKF.S[0,0], self.EKF.S[0,1], 0, 0, 0, self.EKF.S[0,2],
                                    self.EKF.S[1,0], self.EKF.S[1,1], 0, 0, 0, self.EKF.S[1,2],
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    self.EKF.S[2,0], self.EKF.S[2,1], 0, 0, 0, self.EKF.S[2,2]]
                                    
        self.posePub.publish(predPose)
    
    def odometryCallback(self, data):
        odom_x = data.pose.pose.position.x
        odom_y = data.pose.pose.position.y
        odom_quaternion = [
            data.pose.pose.orientation.x, data.pose.pose.orientation.y,
            data.pose.pose.orientation.z, data.pose.pose.orientation.w
        ]
        _, _, odom_yaw = euler_from_quaternion(odom_quaternion)
        odom_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Calculate local difference between 2 odometry readings
        if not hasattr(self, 'last_odom'):
            self.last_odom = np.array([odom_x, odom_y, odom_yaw])
            control = np.array([0.0, 0.0, 0.0])
        else:
            diff_x = odom_x - self.last_odom[0]
            diff_y = odom_y - self.last_odom[1]
            diff_yaw = odom_yaw - self.last_odom[2]
            
            # wrap dyaw
            diff_yaw = (diff_yaw + np.pi) % (2 * np.pi) - np.pi
            
            # Calculate transformation matrix to map global differences to the robot's local frame
            c = cos(self.last_odom[2])
            s = sin(self.last_odom[2])
            
            dx_local = c * diff_x + s * diff_y
            dy_local = -s * diff_x + c * diff_y
            
            control = np.array([dx_local, dy_local, diff_yaw])
            self.last_odom = np.array([odom_x, odom_y, odom_yaw])
        
        if not self.initial:
            self.initial = True
            self.EKF = ExtendedKalmanFilter(odom_x, odom_y, odom_yaw)
        else:
            # Update error covariance mapping rows/cols 0(x), 1(y), and 5(yaw)
            scale_R = 5.0
            self.EKF.R = np.array([
                [odom_covariance[0,0], odom_covariance[0,1], odom_covariance[0,5]],
                [odom_covariance[1,0], odom_covariance[1,1], odom_covariance[1,5]],
                [odom_covariance[5,0], odom_covariance[5,1], odom_covariance[5,5]]
            ]) * scale_R
            self.EKF.predict(u = control)

        self.predictPublish()
        
    def gpsCallback(self, data):
        gps_x = data.pose.pose.position.x
        gps_y = data.pose.pose.position.y
        gps_covariance = np.array(data.pose.covariance).reshape(6, 6)
        
        # Use GPS directly (2D Measurement)
        measurement = np.array([gps_x, gps_y])
        
        if not self.initial:
            self.EKF = ExtendedKalmanFilter(gps_x, gps_y, 0.0)
            self.initial = True
        else:
            # Update error covariance for the 2D measurement
            scale_Q = 1.0
            self.EKF.Q = np.array([
                [gps_covariance[0,0], gps_covariance[0,1]],
                [gps_covariance[1,0], gps_covariance[1,1]]
            ]) * scale_Q
            self.EKF.update(z = measurement)
            
        self.predictPublish()
        return
    
    def gtCallback(self, data):
        self.gt_list[0].append(data.pose.pose.position.x)
        self.gt_list[1].append(data.pose.pose.position.y)
        if self.EKF is not None:
            # Append EKF states to lists for plotting
            self.est_list[0].append(self.EKF.pose[0]) 
            self.est_list[1].append(self.EKF.pose[1])
        return

    def plot_path(self):
        plt.figure(figsize=(10, 8))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.plot(self.gt_list[0], self.gt_list[1], alpha=0.25, linewidth=8, label='Groundtruth path')
        plt.plot(self.est_list[0], self.est_list[1], alpha=0.5, linewidth=3, label='Estimation path')
        plt.title("KF fusion odometry result comparison")
        plt.legend()
        
        # 1. Translate the '~' into a real path (e.g., /root/catkin_ws/result)
        save_dir = os.path.expanduser("~/catkin_ws/result")
        
        # 2. Use os.makedirs (with exist_ok=True) instead of os.mkdir
        if not os.path.exists(save_dir):
            print("Directory does not exist, creating:", save_dir)
            os.makedirs(save_dir, exist_ok=True)
            
        # 3. Save the file
        plt.savefig(os.path.join(save_dir, "result.png"))
        plt.show()
        return
    
if __name__ == '__main__':
    rospy.init_node('kf', anonymous=True)
    fusion = Fusion()
    rospy.spin()
    fusion.plot_path()