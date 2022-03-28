import os
import pygame as pg
from Config.settings import *

class DataLoader:
    def __init__(self, current_dir):

        """
        loads objects for pygame
        """
        self.current_dir = current_dir
        #self.screen = self.game.screen
        self.map_full_path = os.path.join(current_dir,'Config', MAP_FILE)
        self.image_paths = {}
        self.image_paths['Red_Tank'] = os.path.join(current_dir,'Images', 'GameImages', RED_PLAYER_IMAGE)
        self.image_paths['Blue_Tank'] = os.path.join(current_dir,'Images', 'GameImages', BLUE_PLAYER_IMAGE)
        

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

    
    def load_images(self):
        images = {}
        for img_name, img_path in self.image_paths.items():
            file_good = os.path.exists(img_path)
            if file_good:
                images[img_name] = pg.image.load(img_path).convert_alpha()
            else:
                raise Exception('image ' + img_name + ' not found')
        return images


    
