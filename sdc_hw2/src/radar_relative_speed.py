#!/usr/bin/env python3
"""
radar_relative_speed.py (純速度過濾版)
- 拔除 ROI 限制，進行全方位感知
- 使用量測到的相對速度與自車絕對速度比較，過濾靜止背景
"""

import numpy as np
import struct
import rospy
import math
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Vector3Stamped

RADAR_TOPIC = "/ars548/radar_front/objects"
MARKER_TOPIC = "/radar_relative_speed/markers"
VELOCITY_TOPIC = "/filter/velocity"

# ==========================================
# 參數設定區
# ==========================================
# 靜止物件的容忍誤差 (單位: m/s)
# 如果 |相對速度 + 自車速度|小於此值，將被視為靜止背景並過濾掉。
# 考慮到雷達雜訊與濾波器的延遲，通常設定在 0.5 到 1.0 之間。
STATIONARY_TOLERANCE = 0.8  
# ==========================================

def read_points_with_fields(msg, field_names):
    DTYPE_MAP = {1: 'b', 2: 'B', 3: 'h', 4: 'H', 5: 'i', 6: 'I', 7: 'f', 8: 'd'}
    field_info = {}
    for f in msg.fields:
        if f.name in field_names:
            fmt = DTYPE_MAP.get(f.datatype)
            if fmt: field_info[f.name] = (f.offset, fmt)

    endian = '>' if msg.is_bigendian else '<'
    data = msg.data
    step = msg.point_step
    n_points = msg.width * msg.height
    result = {name: np.empty(n_points, dtype=np.float64) for name in field_info}

    for i in range(n_points):
        base = i * step
        for name, (offset, fmt) in field_info.items():
            size = struct.calcsize(fmt)
            val = struct.unpack(endian + fmt, data[base + offset: base + offset + size])[0]
            result[name][i] = val
    return result, n_points

class RadarRelativeSpeed:
    def __init__(self):
        rospy.init_node("radar_relative_speed", anonymous=True)
        self.ego_speed = 0.0  
        
        rospy.Subscriber(VELOCITY_TOPIC, Vector3Stamped, self.ego_speed_callback)
        self.marker_pub = rospy.Publisher(MARKER_TOPIC, MarkerArray, queue_size=1)
        rospy.Subscriber(RADAR_TOPIC, PointCloud2, self.radar_callback)
        
        self.vrel_field = None  
        self.vrel_candidates = ["dynamicsrelvelx", "rangerate", "vrel_rad", "vrelrad", "v_rel_rad", "velocity", "doppler", "range_rate", "vr"]
        
        self.active_object_ids = set()

    def ego_speed_callback(self, msg):
        # 取得自車的絕對速度
        self.ego_speed = math.hypot(msg.vector.x, msg.vector.y)

    def _resolve_vrel_field(self, msg):
        available = [f.name.lower() for f in msg.fields]
        for candidate in self.vrel_candidates:
            if candidate in available:
                for f in msg.fields:
                    if f.name.lower() == candidate:
                        self.vrel_field = f.name
                        rospy.loginfo(f"成功找到速度欄位: {self.vrel_field}")
                        return
        self.vrel_field = "__NONE__"

    def radar_callback(self, msg):
        if self.vrel_field is None:
            self._resolve_vrel_field(msg)
        if self.vrel_field == "__NONE__":
            return

        fields_to_read = ["x", "y", "z", self.vrel_field, "ID"]
        data, n_points = read_points_with_fields(msg, fields_to_read)

        if n_points == 0:
            return

        x = data.get("x", np.array([]))
        y = data.get("y", np.array([]))
        z = data.get("z", np.array([]))
        vrel = data.get(self.vrel_field, np.array([]))
        obj_ids = data.get("ID", np.array([]))

        if len(x) == 0:
            return

        # ---- 核心過濾邏輯 ----
        # 1. 計算絕對速度: 如果測到的 vrel 與 ego_speed 數值接近、方向相反，加起來就會趨近於 0
        v_abs = vrel + self.ego_speed
        
        # 2. 過濾靜止背景: 只保留絕對速度大於容忍誤差的物件 (真正有在移動的物件)
        dynamic_mask = np.abs(v_abs) > STATIONARY_TOLERANCE
        
        x = x[dynamic_mask]
        y = y[dynamic_mask]
        z = z[dynamic_mask]
        vrel = vrel[dynamic_mask]
        
        if len(obj_ids) > 0:
            current_ids = obj_ids[dynamic_mask]
        else:
            current_ids = np.arange(len(x))

        marker_array = MarkerArray()
        stamp = msg.header.stamp
        frame_id = msg.header.frame_id
        
        current_frame_ids = set()

        # 建立並發布 Marker
        for i in range(len(x)):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = "radar_speed"
            
            obj_id = int(current_ids[i])
            marker.id = obj_id
            current_frame_ids.add(obj_id)
            
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD

            marker.pose.position.x = float(x[i])
            marker.pose.position.y = float(y[i])
            marker.pose.position.z = float(z[i]) + 1.5  
            marker.pose.orientation.w = 1.0
            marker.scale.z = 1.0  

            rel_speed = float(vrel[i])
            if rel_speed < 0:
                marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
            marker.color.a = 1.0

            marker.text = "{:.1f} m/s".format(rel_speed)
            marker.lifetime = rospy.Duration(0.5) 
            marker_array.markers.append(marker)

        # 刪除消失的物件
        missing_ids = self.active_object_ids - current_frame_ids
        for missing_id in missing_ids:
            del_marker = Marker()
            del_marker.header.frame_id = frame_id
            del_marker.ns = "radar_speed"
            del_marker.id = missing_id
            del_marker.action = Marker.DELETE
            marker_array.markers.append(del_marker)

        self.active_object_ids = current_frame_ids
        
        if len(marker_array.markers) > 0:
            self.marker_pub.publish(marker_array)

def main():
    node = RadarRelativeSpeed()
    rospy.spin()

if __name__ == "__main__":
    main()