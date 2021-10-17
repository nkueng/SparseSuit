"""
A test script that crawls a directory for config files.
"""
import time
import os


if __name__ == "__main__":

    src_dir = "/media/nic/ExtremeSSD/thesis_data/training_data/Synthetic"
    config_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".yaml"):
                config_files.append(file)
                # ignore the other directories in this root folder
                dirs[:] = []

    print(len(config_files))
