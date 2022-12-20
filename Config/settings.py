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
    'muzzle_flash' : 'shotThin.png',
    'goal' : 'explosion1.png',
    'ammo' : 'crateWood.png',
    'health' : 'barrelGreen_side.png'
}

# game settings
FPS = 60
TITLE = "Tank Game"
MAP_FILE = 'map.txt'
BACKGROUND_COLOR = DARKGREY
GRID_COLOR = LIGHTGREY
TILESIZE = 24

# Player settings
PLAYER_SPEED = 200.0
PLAYER_ROTATION_SPEED = 150.0
PLAYER_HEALTH = 10
PLAYER_BULLETS = 10

# Gun settings
BULLET_SPEED = 500
BULLET_LIFETIME = 1000
BULLET_RATE_DELAY = 500
BULLET_DAMAGE = 1
KICKBACK = 50
BARREL_OFFSET = vec(30, 0)
EXPLOSION_TIME = 1000
MUZZLE_FLASH_SIZE = (20, 10)
FLASH_DURATION = 40

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
MOB_BULLETS = 10

# Goal settings
GOAL_SPAWN_MIN_DIST = 100

# Item settings
AMMO_BULLETS = 5
#AMMO_SPAWN_DIST = 100
AMMO_RESPAWN_TIME = 8000
HEALTH_ADD = 3
HEALTH_RESPAWN_TIME = 10000

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