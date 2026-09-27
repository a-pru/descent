"""Feature indices of the Amelia-10 scene files (data version a10v08)."""

import numpy as np
from easydict import EasyDict

# Feature order of 'agent_sequences' in the scene files.
SEQ_IDX = EasyDict(
    {
        "Speed": 0,
        "Heading": 1,
        "Lat": 2,
        "Lon": 3,
        "Range": 4,
        "Bearing": 5,
        "x": 6,
        "y": 7,
        "z": 8,
    }
)

XYZ = np.zeros(shape=len(SEQ_IDX)).astype(bool)
XYZ[SEQ_IDX.x] = XYZ[SEQ_IDX.y] = XYZ[SEQ_IDX.z] = True

# Selects x, y, z from the ego-frame sequences (x, y, z, speed, heading) built by the dataset.
REL_XYZ = [True, True, True, False, False]
