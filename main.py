import pygame as pg
import sys
import os

# Functions in other scripts of this repo
from Tools.sprites import Player, Wall, Mob
from Tools.data_loader import DataLoader
from Config.settings import *

#config_file = 'setup_config.ini'

class Game:
    def __init__(self):
        # Initialize the pygame module which loads objects specific to operating system and hardware
        pg.init()
        self.current_dir = os.path.dirname(__file__)

        # Get map data
        self.dataloader = DataLoader(self.current_dir)
        self.map_data = self.dataloader.load_map()
        self.get_map_dimensions()

        # Set screen
        self.screen = pg.display.set_mode((self.width, self.height))
        pg.display.set_caption(TITLE)

        # Load images
        self.images = self.dataloader.load_images()
        
        # Set clock
        self.clock = pg.time.Clock()


    def get_map_dimensions(self):
        self.gridwidth = int(len(self.map_data[0].strip('\n')))
        self.gridheight = int(len(self.map_data))
        self.width = self.gridwidth * TILESIZE
        self.height = self.gridheight * TILESIZE


    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.mobs = pg.sprite.Group()
        self.open_spaces = []
        for row, tiles in enumerate(self.map_data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Wall(self, col, row)
                if tile == 'P':
                    self.player = Player(self, col, row)
                if tile == 'M':
                    Mob(self, col, row)
                else:
                    self.open_spaces.append([row, col])


    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.dt = self.clock.tick(FPS) / 1000
            self.events()
            self.update()
            self.draw()


    def quit(self):
        pg.quit()
        sys.exit()


    def update(self):
        # update portion of the game loop
        self.all_sprites.update()


    def draw_grid(self):
        for x in range(0, self.width, TILESIZE):
            pg.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, TILESIZE):
            pg.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y))


    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        self.all_sprites.draw(self.screen)
        pg.display.flip()


    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()


if __name__ == '__main__':
    # create the game object
    g = Game()
    while True:
       g.new()
       g.run()