import os
import pygame as pg
from Config.settings import *

class DataLoader:
    def __init__(self, current_dir):

        """
        loads objects for pygame
        """
        self.current_dir = current_dir
        self.map_full_path = os.path.join(current_dir,'Config', MAP_FILE)
        self.image_dir = os.path.join(current_dir,'Images', 'GameImages')
        

    def load_map(self):
        file_good = os.path.exists(self.map_full_path)
        if file_good:
            map_data = []
            with open(self.map_full_path, 'rt') as f:
                for line in f:
                    map_data.append(line)
            return map_data
        else:
            raise Exception("cannot find map file")

    
    def load_images(self, image_dict):
        images = {}
        for img_name, img_path in image_dict.items():
            file_path = os.path.join(self.image_dir, img_path)
            file_good = os.path.exists(file_path)
            if file_good:
                images[img_name] = pg.image.load(file_path).convert_alpha()
            else:
                raise Exception('image ' + img_name + ' not found')

        return images


    
