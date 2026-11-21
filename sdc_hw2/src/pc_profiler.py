#!/usr/bin/env python3
"""
pc_profiler.py — 這是一個 ROS 節點，用來訂閱 PointCloud2 topic
並印出每一幀的統計資訊 (包含：Topic 名稱、時間戳記、點雲數量、X/Y/Z 軸的最小-最大範圍)。
"""

import sys
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2 

class PCProfiler:
    def __init__(self, topic):
        self.topic = topic
        rospy.init_node("pc_profiler", anonymous=True)
        
        rospy.Subscriber(topic, PointCloud2, self.callback)
        rospy.loginfo("Subscribed to %s  --  waiting for data ...", topic)

    def callback(self, msg):
        """
        當收到 PointCloud2 訊息時觸發此函式。
        """
        
        # field_names=("x", "y", "z"): get position
        points = np.array(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )

        stamp = msg.header.stamp.to_sec()
        n_points = len(points)
        if n_points == 0:
            rospy.logwarn("[%s] Received empty point cloud at t=%.3f", self.topic, stamp)
            return

        
        # get max and min of x, y, z
        x_min, y_min, z_min = points.min(axis=0)
        x_max, y_max, z_max = points.max(axis=0)

        print("=" * 60)
        print("Topic     : {}".format(self.topic))
        print("Timestamp : {:.3f}".format(stamp))
        print("Points    : {}".format(n_points))
        
        print("X range   : [{:.1f}, {:.1f}] m".format(x_min, x_max))
        print("Y range   : [{:.1f}, {:.1f}] m".format(y_min, y_max))
        print("Z range   : [{:.1f}, {:.1f}] m".format(z_min, z_max))
        print("=" * 60)
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 pc_profiler.py <topic_name>")
        print("  e.g. python3 pc_profiler.py /ouster/top_122219002200")
        sys.exit(1)

    # get topic name
    topic = sys.argv[1]
    
    # create profiler
    profiler = PCProfiler(topic)
    
    # keep running
    rospy.spin()


if __name__ == "__main__":
    main()