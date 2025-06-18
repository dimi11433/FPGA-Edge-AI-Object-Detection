import os
import numpy as np

class YoloCalibrationDataReader:
    def __init__(self, calib_data_folder):
        self.data_folder = calib_data_folder
        self.input_name = "images"  
        self.preload()

    def preload(self):
        self.enum_data_dicts = []
        for file in sorted(os.listdir(self.data_folder)):
            if file.endswith(".npy"):
                data = np.load(os.path.join(self.data_folder, file))
                self.enum_data_dicts.append({self.input_name: data})
        self.data_iter = iter(self.enum_data_dicts)

    def get_next(self):
        return next(self.data_iter, None)
