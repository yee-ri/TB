#!/usr/bin/env python3
"""Bounded, subscription-only checks for the NUC's real TB sensor inputs.

This program never publishes a ROS message, changes a parameter, starts a
driver, or moves the robot. A TF lookup proves connectivity, not calibration.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import threading
import time

CAMERA = "/camera/color/image_raw"
DEPTH = "/camera/depth/image_rect_raw"
LIDAR = "/livox/lidar"
ODOM = "/odom"
EXPECTED = {CAMERA: "sensor_msgs/Image", LIDAR: "sensor_msgs/PointCloud2",
            ODOM: "nav_msgs/Odometry"}
MAX_AGE = 0.30
MAX_FUTURE = 0.06


def finite_numbers(values):
    return all(math.isfinite(float(value)) for value in values)


def summary(values):
    if not values:
        return None
    return {"min": min(values), "median": statistics.median(values),
            "max": max(values)}


def frequencies(times):
    intervals = [b - a for a, b in zip(times, times[1:]) if b > a]
    return {"hz": ((len(times) - 1) / (times[-1] - times[0])
                   if len(times) > 1 and times[-1] > times[0] else None),
            "interval_seconds": summary(intervals)}


def image_data(msg):
    data = bytes(msg.data)
    if not data:
        return {"valid_size": False, "nonzero": False, "varied": False}
    stride = max(1, len(data) // 4096)
    sample = data[::stride]
    return {"valid_size": msg.height > 0 and msg.width > 0 and msg.step > 0
            and len(data) == msg.height * msg.step,
            "bytes": len(data), "sample_byte_range": [min(sample), max(sample)],
            "nonzero": any(sample), "varied": min(sample) != max(sample),
            "sample_sha256": hashlib.sha256(sample).hexdigest()}


def depth_data(msg):
    """Check that a complete 16UC1 frame contains actual depth pixels."""
    data = bytes(msg.data)
    valid_size = (msg.height > 0 and msg.width > 0 and msg.step >= 2 * msg.width
                  and len(data) == msg.height * msg.step)
    result = {"valid_size": valid_size, "valid_encoding": msg.encoding == "16UC1",
              "bytes": len(data), "nonzero": False}
    if valid_size and result["valid_encoding"]:
        # Either byte of a 16-bit pixel is nonzero in either byte order.
        # Exclude row padding, which is not a depth measurement.
        result["nonzero"] = any(any(data[row * msg.step:row * msg.step + 2 * msg.width])
                                for row in range(msg.height))
    return result


def cloud_data(msg):
    total = msg.width * msg.height
    fields = {field.name: field for field in msg.fields}
    valid_size = total > 0 and msg.point_step > 0 and msg.row_step >= msg.width * msg.point_step
    valid_size = valid_size and len(msg.data) == msg.row_step * msg.height
    result = {"points": total, "bytes": len(msg.data), "valid_size": valid_size,
              "fields": sorted(fields), "finite_sample_points": 0,
              "nonzero": False, "varied": False}
    if not valid_size or not all(name in fields for name in ("x", "y", "z")):
        return result
    formats = {1: "b", 2: "B", 3: "h", 4: "H", 5: "i", 6: "I", 7: "f", 8: "d"}
    xyz = []
    prefix = ">" if msg.is_bigendian else "<"
    for index in range(min(total, 128)):
        point = index * (total - 1) // max(1, min(total, 128) - 1)
        start = (point // msg.width) * msg.row_step + (point % msg.width) * msg.point_step
        values = []
        for name in ("x", "y", "z"):
            field = fields[name]
            if field.datatype not in formats or field.count < 1:
                return result
            fmt = prefix + formats[field.datatype]
            if field.offset < 0 or field.offset + struct.calcsize(fmt) > msg.point_step:
                return result
            values.append(struct.unpack_from(fmt, msg.data, start + field.offset)[0])
        if finite_numbers(values):
            xyz.append(values)
    result["finite_sample_points"] = len(xyz)
    if xyz:
        result["xyz_sample_bounds"] = [[min(p[i] for p in xyz), max(p[i] for p in xyz)]
                                        for i in range(3)]
        result["nonzero"] = any(any(abs(v) > 1e-6 for v in point) for point in xyz)
        result["varied"] = any(low != high for low, high in result["xyz_sample_bounds"])
    return result


def odom_data(msg):
    pose = msg.pose.pose
    q = pose.orientation
    quaternion = [q.x, q.y, q.z, q.w]
    position = [pose.position.x, pose.position.y, pose.position.z]
    twist = {"linear": [msg.twist.twist.linear.x, msg.twist.twist.linear.y,
                        msg.twist.twist.linear.z],
             "angular": [msg.twist.twist.angular.x, msg.twist.twist.angular.y,
                         msg.twist.twist.angular.z]}
    norm = math.sqrt(sum(v * v for v in quaternion))
    return {"position": position, "quaternion_xyzw": quaternion,
            "quaternion_norm": norm,
            "yaw_radians": math.atan2(2 * (q.w * q.z + q.x * q.y),
                                      1 - 2 * (q.y * q.y + q.z * q.z)),
            "twist": twist, "child_frame_id": msg.child_frame_id,
            "finite": finite_numbers(position + quaternion + twist["linear"] + twist["angular"]),
            "valid_quaternion": abs(norm - 1.0) < 0.01}


def save_report(report, output):
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if output:
        destination = Path(output).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)


def diagnose(duration, require_depth=False):
    import rospy
    import rosgraph
    import tf2_ros
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image, PointCloud2
    from tf2_msgs.msg import TFMessage

    expected_topics = dict(EXPECTED)
    if require_depth:
        expected_topics[DEPTH] = "sensor_msgs/Image"
    rospy.init_node("nuc_diagnostics", anonymous=True, disable_signals=True, disable_rosout=True)
    report = {"status": "unverified", "duration_seconds": duration,
              "require_depth": require_depth,
              "wall_time_unix": time.time(), "read_only": True,
              "limits_seconds": {"maximum_message_age": MAX_AGE,
                                  "maximum_future_stamp": MAX_FUTURE,
                                  "tf_lookup_timeout": 0.0},
              "failures": [], "unverified": [
                  "TF reachability does not independently measure physical mount positions; consult the confirmed survey.",
                  "Odometry movement direction, distance scale, and physical braking are unverified.",
                  "Firmware command watchdog and hardware stop are not proven by stationary subscriptions."]}
    master = rosgraph.Master(rospy.get_name())

    def graph():
        pubs, subs, _ = master.getSystemState()
        types = dict(master.getPublishedTopics("/"))
        return {"published_topics": types,
                "cmd_vel": {"type": types.get("/cmd_vel"),
                            "publishers": dict(pubs).get("/cmd_vel", []),
                            "subscribers": dict(subs).get("/cmd_vel", [])},
                "optional_topics": {name: {"type": types.get(name),
                                           "publishers": dict(pubs).get(name, []),
                                           "subscribers": dict(subs).get(name, [])}
                                    for name in ("/firmware_version", "/version_info", "/motor_power", "/sensor_state")}}

    report["graph_before"] = graph()
    report["use_sim_time"] = {"explicitly_set": rospy.has_param("/use_sim_time"),
                              "value": rospy.get_param("/use_sim_time", False)}
    if report["use_sim_time"]["value"] is not False:
        report["failures"].append("/use_sim_time must be false for real hardware.")
    for topic, expected_type in expected_topics.items():
        actual = report["graph_before"]["published_topics"].get(topic)
        if actual != expected_type:
            report["failures"].append("{}: expected {}, found {}".format(topic, expected_type, actual))
    if report["graph_before"]["cmd_vel"]["publishers"]:
        report["failures"].append("Existing /cmd_vel publishers are active during stationary preparation: "
                                  + ", ".join(report["graph_before"]["cmd_vel"]["publishers"]))

    buffer = tf2_ros.Buffer(cache_time=rospy.Duration(15.0))
    listener = tf2_ros.TransformListener(buffer)
    lock = threading.Lock()
    samples = {topic: [] for topic in expected_topics}
    authorities = {}
    callback_errors = []

    def tf_callback(msg, topic):
        caller = getattr(msg, "_connection_header", {}).get("callerid", "unknown")
        with lock:
            for transform in msg.transforms:
                child = transform.child_frame_id.lstrip("/")
                parent = transform.header.frame_id.lstrip("/")
                authorities.setdefault(child, set()).add((parent, caller, topic))

    def callback(msg, topic):
        try:
            now = time.time()
            stamp = msg.header.stamp.to_sec()
            entry = {"arrival_monotonic": time.monotonic(), "stamp_seconds": stamp,
                     "age_at_receipt_seconds": now - stamp,
                     "frame_id": msg.header.frame_id,
                     "stamp": msg.header.stamp}
            if topic in (CAMERA, DEPTH):
                entry.update({"resolution": [msg.width, msg.height], "encoding": msg.encoding,
                              "step": msg.step,
                              "data": depth_data(msg) if topic == DEPTH else image_data(msg)})
            elif topic == LIDAR:
                entry["data"] = cloud_data(msg)
            else:
                entry["data"] = odom_data(msg)
            with lock:
                samples[topic].append(entry)
        except Exception as error:
            with lock:
                callback_errors.append("{}: {}: {}".format(topic, type(error).__name__, error))

    topic_classes = [(CAMERA, Image), (LIDAR, PointCloud2), (ODOM, Odometry)]
    if require_depth:
        topic_classes.append((DEPTH, Image))
    subscriptions = [rospy.Subscriber(topic, kind, callback, callback_args=topic,
                                      queue_size=5, buff_size=16 * 1024 * 1024)
                     for topic, kind in topic_classes]
    subscriptions += [rospy.Subscriber(topic, TFMessage, tf_callback, callback_args=topic,
                                       queue_size=100) for topic in ("/tf", "/tf_static")]
    start = time.monotonic()
    deadline = start + duration
    while time.monotonic() < deadline and not rospy.is_shutdown():
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    # Allow the TF subscriber to deliver transforms already received during the
    # bounded observation; measured-time lookups themselves have zero timeout.
    report["observation_elapsed_seconds"] = time.monotonic() - start
    with lock:
        captured = {topic: list(entries) for topic, entries in samples.items()}
        authority_snapshot = {child: sorted(entries) for child, entries in authorities.items()}
        report["callback_errors"] = list(callback_errors)
    report["failures"].extend(callback_errors)
    report["tf_authorities"] = {}
    for child, entries in authority_snapshot.items():
        parents = sorted({item[0] for item in entries})
        callers = sorted({item[1] for item in entries})
        report["tf_authorities"][child] = {"parents": parents, "callerids": callers,
                                          "sources": [{"parent": p, "callerid": c, "topic": t}
                                                      for p, c, t in entries],
                                          "duplicate_authority": len(callers) > 1 or len(parents) > 1}
        if len(callers) > 1 or len(parents) > 1:
            report["failures"].append("TF child {} has multiple authorities or parents: {}".format(child, entries))
    report["topics"] = {}
    report["measured_time_tf"] = []
    for topic, entries in captured.items():
        details = {"expected_type": expected_topics[topic], "messages_received": len(entries),
                   "status": "unverified"}
        report["topics"][topic] = details
        if len(entries) < 2:
            report["failures"].append("{}: fewer than two actual messages received.".format(topic))
            continue
        details.update(frequencies([entry["arrival_monotonic"] for entry in entries]))
        details["stamp_rate"] = frequencies([entry["stamp_seconds"] for entry in entries])
        details["frame_ids"] = sorted({entry["frame_id"] for entry in entries})
        ages = [entry["age_at_receipt_seconds"] for entry in entries]
        details["age_at_receipt_seconds"] = summary(ages)
        details["age_of_last_message_now_seconds"] = time.time() - entries[-1]["stamp_seconds"]
        details["zero_stamps"] = sum(entry["stamp_seconds"] <= 0 for entry in entries)
        details["non_increasing_stamps"] = sum(b["stamp_seconds"] <= a["stamp_seconds"]
                                                for a, b in zip(entries, entries[1:]))
        last = {key: value for key, value in entries[-1].items()
                if key not in ("stamp", "arrival_monotonic")}
        details["latest"] = last
        topic_failures = []
        if details["zero_stamps"] or details["non_increasing_stamps"]:
            topic_failures.append("zero or non-increasing measurement timestamps")
        if min(ages) < -MAX_FUTURE or max(ages) > MAX_AGE:
            topic_failures.append("message age/future stamp exceeds 0.30s/0.06s limits")
        if details["age_of_last_message_now_seconds"] > MAX_AGE:
            topic_failures.append("last message has become stale")
        if "" in details["frame_ids"] or len(details["frame_ids"]) != 1:
            topic_failures.append("empty or changing frame_id")
        if topic == CAMERA:
            details["distinct_image_samples"] = len({e["data"].get("sample_sha256") for e in entries})
            if not 25.0 <= details["hz"] <= 35.0:
                topic_failures.append("color image arrival frequency is outside approximately 30Hz (25–35Hz)")
            if any(not e["data"]["valid_size"] or not e["data"]["nonzero"]
                   or not e["data"]["varied"] for e in entries):
                topic_failures.append("image data size, nonzero pixels, or spatial variation failed")
            if last["encoding"] not in ("rgb8", "bgr8", "rgba8", "bgra8"):
                topic_failures.append("image encoding is not a supported real color encoding")
        elif topic == DEPTH:
            if any(not e["data"]["valid_size"] or not e["data"]["valid_encoding"]
                   or not e["data"]["nonzero"] for e in entries):
                topic_failures.append("depth requires valid 16UC1 data and nonzero depth pixels")
        elif topic == LIDAR:
            details["point_counts"] = summary([e["data"]["points"] for e in entries])
            if any(not e["data"]["valid_size"] or not e["data"]["nonzero"]
                   or not e["data"]["varied"] or not e["data"]["finite_sample_points"] for e in entries):
                topic_failures.append("PointCloud2 is empty, malformed, or lacks varied finite XYZ data")
        else:
            if any(not e["data"]["finite"] or not e["data"]["valid_quaternion"] for e in entries):
                topic_failures.append("odometry pose/twist or quaternion is invalid")
            if details["frame_ids"] != ["odom"] or any(e["data"]["child_frame_id"] != "base_footprint" for e in entries):
                topic_failures.append("odometry frames must be odom→base_footprint")
        # Use up to ten recent historic samples, excluding the last 0.10s of
        # arrivals. This avoids querying beyond the newest TF at delivery time.
        historic = [e for e in entries if e["arrival_monotonic"] <= deadline - 0.10]
        for entry in historic[-10:]:
            sources = {entry["frame_id"].lstrip("/"), "odom"}
            for source in sorted(sources):
                tf_check = {"topic": topic, "source": source, "target": "base_footprint",
                            "measurement_stamp_seconds": entry["stamp_seconds"], "status": "unverified"}
                try:
                    transform = buffer.lookup_transform("base_footprint", source, entry["stamp"], rospy.Duration(0.0))
                    tf_check["status"] = "pass"
                    tf_check["transform_stamp_seconds"] = transform.header.stamp.to_sec()
                    tf_check["translation"] = [transform.transform.translation.x,
                                                transform.transform.translation.y,
                                                transform.transform.translation.z]
                    tf_check["quaternion_xyzw"] = [transform.transform.rotation.x,
                                                    transform.transform.rotation.y,
                                                    transform.transform.rotation.z,
                                                    transform.transform.rotation.w]
                except Exception as error:
                    tf_check["error"] = "{}: {}".format(type(error).__name__, error)
                    topic_failures.append("measured-time TF {}→base_footprint failed".format(source))
                report["measured_time_tf"].append(tf_check)
        if not historic:
            topic_failures.append("no historic measurement available for TF lookup")
        details["failures"] = sorted(set(topic_failures))
        details["status"] = "pass" if not topic_failures else "unverified"
        report["failures"].extend("{}: {}".format(topic, failure) for failure in details["failures"])
    report["graph_after"] = graph()
    if report["graph_after"]["cmd_vel"]["publishers"]:
        report["failures"].append("/cmd_vel publishers appeared during stationary diagnostics: "
                                  + ", ".join(report["graph_after"]["cmd_vel"]["publishers"]))
    report["failures"] = sorted(set(report["failures"]))
    report["status"] = "empirical_sensor_tf_pass" if not report["failures"] else "unverified"
    for subscription in subscriptions:
        subscription.unregister()
    # Keep listener referenced throughout the full observation and all lookups.
    del listener
    rospy.signal_shutdown("Read-only diagnostics complete")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0,
                        help="Wall-clock observation seconds, bounded to 3–30 (default: 12)")
    parser.add_argument("--output", help="Save the diagnostic JSON to this path")
    parser.add_argument("--require-depth", action="store_true",
                        help="Require fresh 16UC1 depth images for full-course gate detection")
    args = parser.parse_args()
    if not 3 <= args.duration <= 30:
        parser.error("--duration must be between 3 and 30 seconds")
    try:
        report = diagnose(args.duration, require_depth=args.require_depth)
    except Exception as error:
        report = {"status": "unverified", "read_only": True,
                  "failures": ["{}: {}".format(type(error).__name__, error)]}
    # Invalid numeric data is already a failure; preserve its presence without
    # writing non-standard NaN/Infinity JSON.
    def sanitize(value):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value
    save_report(sanitize(report), args.output)
    return 0 if report["status"] == "empirical_sensor_tf_pass" else 1


if __name__ == "__main__":
    sys.exit(main())
