# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The hazcam obstacle message: a sensor_msgs/PointCloud2 in odom, packed and unpacked here.

A standard message rather than a custom one so RViz can show it as is. Fields are
x, y, z (float32, z = 0: the costmap is 2D) and `step` (float32, the step height in
metres that made the cell an obstacle, for diagnosis). Both ends live in this one
module so the layout cannot drift between producer and consumer.
"""

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import PointField

_FIELDS = ("x", "y", "z", "step")
_DTYPE = np.dtype([(name, "<f4") for name in _FIELDS])


def make_cloud(xy: np.ndarray, step_m: np.ndarray, frame_id: str, stamp) -> PointCloud2:
    """Pack obstacle (x, y) and step heights into a PointCloud2."""
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    data = np.zeros(len(xy), dtype=_DTYPE)
    data["x"], data["y"] = xy[:, 0], xy[:, 1]
    data["step"] = np.asarray(step_m, dtype=np.float64).reshape(-1)
    msg = PointCloud2()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp
    msg.height = 1
    msg.width = len(xy)
    msg.fields = [
        PointField(name=name, offset=4 * i, datatype=PointField.FLOAT32, count=1)
        for i, name in enumerate(_FIELDS)
    ]
    msg.is_bigendian = False
    msg.point_step = _DTYPE.itemsize
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = True
    msg.data = data.tobytes()
    return msg


def xy_from_cloud(msg: PointCloud2) -> np.ndarray:
    """(N, 2) float64 x, y from any PointCloud2 with float32 x and y fields."""
    offsets = {f.name: f.offset for f in msg.fields if f.datatype == PointField.FLOAT32}
    if "x" not in offsets or "y" not in offsets:
        raise ValueError("point cloud has no float32 x and y fields")
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 2))
    order = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({"names": ["x", "y"], "formats": [order + "f4"] * 2,
                      "offsets": [offsets["x"], offsets["y"]], "itemsize": msg.point_step})
    raw = np.frombuffer(bytes(msg.data), dtype=dtype, count=n)
    return np.column_stack([raw["x"], raw["y"]]).astype(np.float64)
