"""
Converts directories of IMU sensor data (SREC) and corresponding poses (FBX) to the format we use.
"""
import os
import numpy as np
import SrecReader
from scipy.spatial.transform import Rotation as R
from sparsesuit.constants.sensors import SENS_NAMES_SSP, SREC_2_SSP


def get_srec(file):
    reader = SrecReader.SRecReader(filename=file.split(".srec")[0])
    try:
        rec = reader.ReadSrec(parse=False)
    except AssertionError:
        print("Asset {} is not valid.".format(file))
        rec = SrecReader.SRec(".")
    return rec


def parse_srec(files):
    # get sorting order for to our convention
    rec = get_srec(files[0])
    sensor_add = rec.suits[0].frames[0].addresses
    sensor_names = [SREC_2_SSP[rec.setSensorName(add)] for add in sensor_add]
    sort_ids = [sensor_names.index(sensor) for sensor in SENS_NAMES_SSP]

    # parse all files
    accs = []
    oris = []
    for file in files:
        print(file)
        rec = get_srec(file)
        if rec.num_suits != 1:
            continue
        frames = rec.suits[0].frames

        for frame in frames:
            # convert acceleration from g to m/s²
            acc_local = np.array(frame.acceleration)[sort_ids] * 9.81
            acc_local = np.expand_dims(acc_local, axis=2)

            # convert orientations from quaternion to rotation matrix format
            ori = R.from_quat(np.array(frame.quaternion)[sort_ids]).as_matrix()

            # transform acceleration to global frame
            acc_global = np.einsum("abc,abd->abd", ori, acc_local).squeeze()

            oris.append(ori)
            accs.append(acc_global)

    return oris, accs


if __name__ == "__main__":

    # set src directory with files
    src_dir = "/media/nic/ExtremeSSD/real_dataset/SSP_data"

    # walk over all files in directory and collect relevant paths
    srec_files = []
    fbx_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".srec"):
                srec_files.append(os.path.join(root, file))

            if file.endswith(".fbx"):
                fbx_files.append(os.path.join(root, file))

    # parse SREC
    imu_data = parse_srec(srec_files)

    # clean IMU data

    # parse FBX

    # clean pose data
