import pygame as pg
vec = pg.math.Vector2

# define some colors (R, G, B)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGREY = (40, 40, 40)
LIGHTGREY = (100, 100, 100)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

# Images in folder
Tank_images = {
    'Red' : 'tank_red.png',
    'Blue' : 'tank_blue.png',
    'Green' : 'tank_green.png',
    'Sand' : 'tank_sand.png',
    'Dark' : 'tank_dark.png',
}

Bullet_images = {
    'Red' : 'bulletRed1.png',
    'Blue' : 'bulletBlue1.png',
    'Green' : 'bulletGreen1.png',
    'Sand' : 'bulletSand1.png',
    'Dark' : 'bulletDark1.png',
}

Other_images = {
    'explosion' : 'explosion2.png',
    'mine_explosion' : 'explosion4.png',
    'muzzle_flash' : 'shotThin.png',
    'goal' : 'explosion1.png',
    'ammo' : 'crateWood.png',
    'health' : 'HealthPack.png'
}

# game settings
FPS = 50
TITLE = "Tank Game"
MAP_FILE = 'map_simple.txt'
BACKGROUND_COLOR = DARKGREY
GRID_COLOR = LIGHTGREY
TILESIZE = 24
GAME_TIME_LIMIT = 120 # seconds

# Player settings
PLAYER_SPEED = 200.0
PLAYER_ROTATION_SPEED = 200.0
PLAYER_HEALTH = 10
PLAYER_BULLETS = 10
PLAYER_MINES = 5

# Gun settings
BULLET_SPEED = 500
BULLET_LIFETIME = 1 * 1000 # 1 sec * 1000 milliseconds
BULLET_RATE_DELAY = 500
BULLET_DAMAGE = 1
KICKBACK = 50
BARREL_OFFSET = vec(30, 0)
EXPLOSION_TIME = 1000 # 1 sec * 1000 milliseconds
MUZZLE_FLASH_SIZE = (20, 10)
FLASH_DURATION = 40

# Mine settings
MINE_LIFETIME = 5 * 1000 # 5 sec * 1000 milliseconds
MINE_RATE_DELAY = 500
MINE_DAMAGE = 1

# Image settings
RED_PLAYER_INITIAL_ROTATION = 90
BLUE_PLAYER_INITIAL_ROTATION = 90
PLAYER_HIT_RECTANGLE = pg.Rect(0, 0, 45, 45)
COUNTER_OFFSET = vec(30, 0)

# Mob settings
MOB_SPEED = 150
MOB_HEALTH = 5
MOB_HIT_RECTANGLE = pg.Rect(0, 0, 45, 45)
MOB_DAMAGE = 1
MOB_KNOCKBACK = 20
SHOOT_CONE = 5
MOB_BULLETS = 0
MOB_MINES = 0
MOB_UPDATE_DELAY = 500
MOB_AGGRESSIVENESS = 0.5
MOB_MINE_PROB = 0.05

# Goal settings
GOAL_SPAWN_MIN_DIST = 100

# Item settings
AMMO_BULLETS = 5
AMMO_RESPAWN_TIME = 8000 # 8 sec * 1000 milliseconds
HEALTH_ADD = 3
HEALTH_RESPAWN_TIME = 10000 # 10 sec * 1000 milliseconds

# Draw Layers
WALL_LAYER = 1
ITEM_LAYER = 2
PLAYER_LAYER = 3
MOB_LAYER = 3
BULLET_LAYER = 4
EFFECT_LAYER = 5

# Score Settings
FONT_NAME = pg.font.match_font('arial')
FONT_SIZE = 18

# Points in game
GOAL_POINTS = 3
HIT_POINTS = 1