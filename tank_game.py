# Standard Library Imports
import sys
import os
import random

# Conda imports
import pygame as pg

# Functions in other scripts of this repo
from Tools.sprites import Player, Wall, Mob, Goal, Ammo, Health
from Tools.data_loader import DataLoader
from Config.game_settings import *
from Tools.sprites import Explosion
from Tools.helper_methods import collide_hit_rect, draw_text

class Game:
    def __init__(self, show_display=True):
        # Initialize the pygame module which loads objects specific to operating system and hardware
        pg.init()
        self.current_dir = os.path.dirname(__file__)

        # Get map data
        self.dataloader = DataLoader(self.current_dir)
        self.map_data = self.dataloader.load_map()
        self.get_map_dimensions()

        # Set screen
        if not show_display:
            flags = pg.HIDDEN
        else:
            flags = 0
        self.screen = pg.display.set_mode((self.width, self.height), flags)
        pg.display.set_caption(TITLE)

        # Load images
        self.tank_images = self.dataloader.load_images(Tank_images)
        self.bullet_images = self.dataloader.load_images(Bullet_images)
        self.other_images = self.dataloader.load_images(Other_images)
        
        # Set clock
        self.clock = pg.time.Clock()

    def get_map_dimensions(self):
        self.gridwidth = int(len(self.map_data[0].strip('\n')))
        self.gridheight = int(len(self.map_data))
        self.width = self.gridwidth * TILESIZE
        self.height = self.gridheight * TILESIZE

    def new(self):
        # initialize all variables and do all the setup for a new game
        self.all_sprites = pg.sprite.LayeredUpdates()
        self.players = pg.sprite.Group()
        self.walls = pg.sprite.Group()
        self.mobs = pg.sprite.Group()
        self.movers = pg.sprite.Group()
        self.goals = pg.sprite.Group()
        self.ammo_boxes = pg.sprite.Group()
        self.health_kits = pg.sprite.Group()
        self.player_bullets = pg.sprite.Group()
        self.mob_bullets = pg.sprite.Group()
        self.explosion = pg.sprite.Group()
        self.player_mines = pg.sprite.Group()
        self.mob_mines = pg.sprite.Group()
        self.open_pos = []
        self.open_spaces = []
        self.health_locations = []
        self.ammo_locations = []
        for row, tiles in enumerate(self.map_data):
            for col, tile in enumerate(tiles):
                if tile == '1':
                    Wall(self, col, row)
                elif tile == 'P':
                    self.player = Player(self, col, row)
                elif tile == 'M':
                    Mob(self, col, row)
                elif tile == 'G':
                    self.goal = Goal(self, col, row)
                elif tile == 'A':
                    Ammo(self, col, row)
                    self.ammo_locations.append([col, row])
                elif tile == 'H':
                    Health(self, col, row)
                    self.health_locations.append([col, row])
                elif tile =='\n' or tile == '\r\n':
                    continue
                else:
                    self.open_spaces.append([col, row])
                    self.open_pos.append(vec(col, row) * TILESIZE)

        # Set Score
        self.score = {'Blue': 0, 'Red': 0}

        # Set clock
        minutes = GAME_TIME_LIMIT // 60
        seconds = GAME_TIME_LIMIT - (60 * minutes)
        self.minutes = minutes
        self.seconds = seconds
        self.milliseconds = 0
        self.time_countdown = GAME_TIME_LIMIT * 1000 # convert to milliseconds

    def run(self):
        # game loop - set self.playing = False to end the game
        self.playing = True
        while self.playing:
            self.advance_time()
            self.events()
            self.update()
            self.timer()
            self.draw()
        self.game_over()

    def game_over(self):
        game_over_waiting = True
        wait_time = 1000
        #self.draw()
        while game_over_waiting: 
            self.advance_time()
            self.game_over_text()
            self.events()
            wait_time -= self.dt_ms
            if wait_time <= 0:
                game_over_waiting = False

    def game_over_text(self):
        draw_text(self.screen, "GAME OVER", RED, 2 * FONT_SIZE, self.width / 2, self.height / 2, 0)
        pg.display.flip()

    def advance_time(self):
        self.dt_ms = self.clock.tick(FPS)
        self.dt = self.dt_ms / 1000

    def hit_by_bullet(self, hit_sprites):
        for hit_sprite in hit_sprites:
            hit_sprite.health -= BULLET_DAMAGE
            hit_sprite.vel = vec(0,0)

            if hit_sprite in self.players:
                self.score['Red'] += HIT_POINTS
            elif hit_sprite in self.mobs:
                self.score['Blue'] += HIT_POINTS

    def hit_by_mine(self, hit_sprites):
            for hit_sprite in hit_sprites:
                hit_sprite.health -= MINE_DAMAGE
                hit_sprite.vel = vec(0,0)
                Explosion(hit_sprites[hit_sprite][0])

                if hit_sprite in self.players:
                    self.score['Red'] += HIT_POINTS
                elif hit_sprite in self.mobs:
                    self.score['Blue'] += HIT_POINTS

    def hit_goal(self, sprites_on_goal):
        for sprite_on_goal in sprites_on_goal:
            if sprite_on_goal in self.players:
                self.score['Blue'] += GOAL_POINTS
            elif sprite_on_goal in self.mobs:
                self.score['Red'] += GOAL_POINTS
        for goal in sprites_on_goal[sprite_on_goal]:
            # Get possible spawn locations far enough away
            possible_locs = [pos for pos in self.open_spaces if goal.pos.distance_to(vec(pos) * TILESIZE) >= GOAL_SPAWN_MIN_DIST]
            new_location = random.choice(possible_locs)
            # Kill current goal and re-spawn a new goal
            goal.kill()
            Goal(self, new_location[0], new_location[1])

    def hit_ammo(self, sprites_on_ammo):
        for sprite_on_ammo in sprites_on_ammo:
            ammo_needs_respawn = False
            sprite_ammo = sprites_on_ammo[sprite_on_ammo][0]
            if sprite_ammo.available:
                if sprite_on_ammo in self.players and sprite_on_ammo.bullets < PLAYER_BULLETS:
                    sprite_on_ammo.bullets += AMMO_BULLETS
                    ammo_needs_respawn = True
                    if sprite_on_ammo.bullets > PLAYER_BULLETS:
                        sprite_on_ammo.bullets = PLAYER_BULLETS
                if sprite_on_ammo in self.mobs and sprite_on_ammo.bullets < MOB_BULLETS:
                    sprite_on_ammo.bullets += AMMO_BULLETS
                    ammo_needs_respawn = True
                    if sprite_on_ammo.bullets > MOB_BULLETS:
                        sprite_on_ammo.bullets = MOB_BULLETS
                if ammo_needs_respawn:
                    sprite_ammo.hit_time = pg.time.get_ticks()
                    sprite_ammo.available = False
                    sprite_ammo.image.set_alpha(0)

    def hit_health(self, sprites_on_health):
        for sprite_on_health in sprites_on_health:
            health_needs_respawn = False
            sprite_health = sprites_on_health[sprite_on_health][0]
            if sprite_health.available:
                if sprite_on_health in self.players and sprite_on_health.health < PLAYER_HEALTH:
                    sprite_on_health.health += HEALTH_ADD
                    health_needs_respawn = True
                    if sprite_on_health.health > PLAYER_HEALTH:
                        sprite_on_health.health = PLAYER_HEALTH
                if sprite_on_health in self.mobs and sprite_on_health.health < MOB_HEALTH:
                    sprite_on_health.health += HEALTH_ADD
                    health_needs_respawn = True
                    if sprite_on_health.health > MOB_HEALTH:
                        sprite_on_health.health = MOB_HEALTH
                if health_needs_respawn:
                    sprite_health.hit_time = pg.time.get_ticks()
                    sprite_health.available = False
                    sprite_health.image.set_alpha(0)

    def update(self):
        # update portion of the game loop
        self.all_sprites.update()

        # bullets hit mob
        hits = pg.sprite.groupcollide(self.mobs, self.player_bullets, False, True)
        if hits:
            self.hit_by_bullet(hits)
            
        # bullets hit player
        hits = pg.sprite.groupcollide(self.players, self.mob_bullets, False, True)
        if hits:
            self.hit_by_bullet(hits)

        # mob hit mine
        hits = pg.sprite.groupcollide(self.mobs, self.player_mines, False, True)
        if hits:
            self.hit_by_mine(hits)

        # player hit mine
        hits = pg.sprite.groupcollide(self.players, self.mob_mines, False, True)
        if hits:
            self.hit_by_mine(hits)

        # Player or Mob hits goal
        sprites_on_goal = pg.sprite.groupcollide(self.movers, self.goals, False, True)
        if sprites_on_goal:
            self.hit_goal(sprites_on_goal)

        # Player or Mob hits ammo
        sprites_on_ammo = pg.sprite.groupcollide(self.movers, self.ammo_boxes, False, False)
        if sprites_on_ammo:
            self.hit_ammo(sprites_on_ammo)

        # Player or Mob hits health
        sprites_on_health = pg.sprite.groupcollide(self.movers, self.health_kits, False, False)
        if sprites_on_health:
            self.hit_health(sprites_on_health)

        # Player dead   
        if self.player.health <= 0:
            Explosion(self.player)
            self.playing = False

        # if all enemy killed, end game
        if not self.mobs:
            self.playing = False

    def draw_grid(self):
        for x in range(0, self.width, TILESIZE):
            pg.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self.height))
        for y in range(0, self.height, TILESIZE):
            pg.draw.line(self.screen, GRID_COLOR, (0, y), (self.width, y))

    def draw(self):
        self.screen.fill(BACKGROUND_COLOR)
        self.draw_grid()
        #self.all_sprites.draw(self.screen)
        for sprite in self.all_sprites:
            if isinstance(sprite, Mob) or isinstance(sprite, Player):
                sprite.draw_health()
                #sprite.draw_bullet_counter()
                #sprite.draw_mine_counter()
        self.all_sprites.draw(self.screen)

        # draw score string at top
        score_string = f"Blue: {self.score['Blue']}  Red: {self.score['Red']}"
        draw_text(self.screen, score_string, BLACK, FONT_SIZE, self.width / 3, 1, 0)

        # draw timer at top
        time_string = f"Minutes: {self.minutes} Seconds: {self.seconds}"
        draw_text(self.screen, time_string, BLACK, FONT_SIZE, 2 * self.width / 3, 1, 0)

        # Blue ammo display
        blue_ammo_string = f"BLUE - Bullets: {self.player.bullets} Mines: {self.player.mines}"
        draw_text(self.screen, blue_ammo_string, BLACK, FONT_SIZE, 0.2 * self.width, 0.96 * self.height, 0)

        # Red ammo display
        min_bullets = MOB_BULLETS
        min_mines = MOB_MINES
        for sprite in self.mobs.sprites():
            if sprite.bullets < min_bullets:
                min_bullets = sprite.bullets
            if sprite.mines < min_mines:
                min_mines = sprite.mines
        red_ammo_string = f"RED - Bullets: {min_bullets} Mines: {min_mines}"
        draw_text(self.screen, red_ammo_string, BLACK, FONT_SIZE, 0.8 * self.width, 0.96 * self.height, 0)

        pg.display.flip()

    def timer(self):
        if self.milliseconds > 1000:
            self.seconds -= 1
            self.milliseconds -= 1000
        if self.seconds < 0:
            self.minutes -= 1
            self.seconds = 60

        self.time_countdown -= self.dt_ms
        self.milliseconds += self.dt_ms
        if self.time_countdown <= 0:
            self.playing = False

    def events(self):
        # catch all events here
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.quit()
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.quit()

    def quit(self):
        pg.quit()
        sys.exit()


if __name__ == '__main__':
    # create the game object
    g = Game()
    while True:
       g.new()
       g.run()