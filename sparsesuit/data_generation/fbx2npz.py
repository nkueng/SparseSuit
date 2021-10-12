import os
import numpy as np
from sparsesuit.constants import paths
from Skynet.rkk_io.MLDataImportClass import FbxImporter

# get list of all fbx files in source directory
src_dir = os.path.join(paths.DATA_PATH, "raw_SSP_dataset/SSP_data/Export")

# walk over all files in directory and collect relevant paths
fbx_folders = []
for root, dirs, files in os.walk(src_dir):
    for file in files:
        if file.endswith(".fbx"):
            fbx_folders.append(root)
            break

fbx_folders = sorted(fbx_folders)

# convert all fbx iteratively
for fbx_folder in fbx_folders:
    fbx_importer = FbxImporter(source_dir=fbx_folder, fps_=100)
    fbx_importer(do_trim=False)

    for i, used_file_i in enumerate(fbx_importer.used_files):
        skeleton = fbx_importer.skeletons[i]
        parents = fbx_importer.parents[i]
        positions = fbx_importer.p_parent[i]
        rotations = fbx_importer.r_parent[i]

        fbx_dict = {
            "skeleton": skeleton,
            "parents": parents,
            "positions": positions,
            "rotations": rotations,
        }

        # drop motion data into npz
        file_name = used_file_i.split(".")[0]
        file_path = os.path.join(fbx_folder, file_name) + ".npz"
        with open(file_path, "wb") as fout:
            np.savez_compressed(fout, **fbx_dict)

        # remove all intermediate files except for fbx
        all_files = os.listdir(fbx_folder)
        for file in all_files:
            if file.endswith(".fbx") or file.endswith(".npz"):
                continue
            if file_name in file:
                try:
                    file_path = os.path.join(fbx_folder, file)
                    os.remove(file_path)
                except OSError:
                    pass



