# Standard Library Imports
import random

# Conda imports
import pygame as pg

# Functions in other scripts of this repo
from Config.settings import *
from Tools.helper_methods import collide_hit_rect, draw_text
from Tools.A_Star import A_Star

vec = pg.math.Vector2

def collide_with_walls(sprite, group, dir):
    if dir == 'x':
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            if hits[0].rect.centerx > sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.left - sprite.hit_rect.width / 2
            if hits[0].rect.centerx < sprite.hit_rect.centerx:
                sprite.pos.x = hits[0].rect.right + sprite.hit_rect.width / 2
            sprite.vel.x = 0
            sprite.hit_rect.centerx = sprite.pos.x
    if dir == 'y':
        hits = pg.sprite.spritecollide(sprite, group, False, collide_hit_rect)
        if hits:
            if hits[0].rect.centery > sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.top - sprite.hit_rect.height / 2
            if hits[0].rect.centery < sprite.hit_rect.centery:
                sprite.pos.y = hits[0].rect.bottom + sprite.hit_rect.height / 2
            sprite.vel.y = 0
            sprite.hit_rect.centery = sprite.pos.y

def draw_health_box(sprite, full_health):
    if sprite.health > 0.6 * full_health:
        col = GREEN
    elif sprite.health > 0.3 * full_health:
        col = YELLOW
    else:
        col = RED
    width = int(sprite.hit_rect.width * sprite.health / full_health)
    sprite.health_bar = pg.Rect(0, 40, width, 5)
    pg.draw.rect(sprite.image, col, sprite.health_bar)
    # if sprite.health < full_health:
    #     pg.draw.rect(sprite.image, col, sprite.health_bar)

# def draw_bullet_number(sprite, full_bullets):
#     if sprite.bullets > 0.6 * full_bullets:
#         col = GREEN
#     elif sprite.bullets > 0.3 * full_bullets:
#         col = YELLOW
#     else:
#         col = RED
#     bullet_count = str(sprite.bullets)
#     pos = vec(sprite.image.get_width() / 2, sprite.image.get_height() / 2) - vec(sprite.image.get_width() * 0.3, 0).rotate(-sprite.rot)
#     draw_text(sprite.image, bullet_count, col, 14, pos[0], pos[1], 90 + sprite.rot)

# def draw_mine_number(sprite, full_mines):
#     if sprite.mines > 0.6 * full_mines:
#         col = GREEN
#     elif sprite.mines > 0.3 * full_mines:
#         col = YELLOW
#     else:
#         col = RED
#     mine_count = str(sprite.mines)
#     pos = vec(sprite.image.get_width() / 2, sprite.image.get_height() / 2) - vec(sprite.image.get_width() * 0.3, 0).rotate(-sprite.rot)
#     draw_text(sprite.image, mine_count, col, 14, pos[0], pos[1], 270 + sprite.rot)

def shoot_bullet(sprit):
    now = pg.time.get_ticks()
    if (now - sprit.last_shot > BULLET_RATE_DELAY) and (sprit.bullets > 0):
        sprit.last_shot = now
        dir = vec(1, 0).rotate(-sprit.rot)
        pos = sprit.pos + BARREL_OFFSET.rotate(-sprit.rot)
        Bullet(sprit, pos, dir)
        sprit.vel = vec(-KICKBACK, 0).rotate(-sprit.rot)
        MuzzleFlash(sprit, pos)
        sprit.bullets -= 1

def lay_mine(sprit):
    now = pg.time.get_ticks()
    if (now - sprit.last_shot > MINE_RATE_DELAY) and (sprit.mines > 0):
        sprit.last_shot = now
        Mine(sprit)
        sprit.mines -= 1

def route_to_closets_ammo(sprite):
    ammo_boxes_dists = {}
    for ammo_box in sprite.game.ammo_boxes:
        if ammo_box.available:
            dist = sprite.pos.distance_to(ammo_box.pos)
        else:
            dist = float('inf')
        ammo_boxes_dists[dist] = ammo_box
    ammo_box_to_move = ammo_boxes_dists[min(ammo_boxes_dists)]
    route = sprite.A_Star.find_route(sprite.pos, ammo_box_to_move.pos)
    sprite.route = route

def route_to_closets_health(sprite):
    health_kit_dists = {}
    for health_kit in sprite.game.health_kits:
        if health_kit.available:
            dist = sprite.pos.distance_to(health_kit.pos)
        else:
            dist = float('inf')
        health_kit_dists[dist] = health_kit
    health_kit_to_move = health_kit_dists[min(health_kit_dists)]
    route = sprite.A_Star.find_route(sprite.pos, health_kit_to_move.pos)
    sprite.route = route

def route_to_target_sprite(sprite, target):
    route = sprite.A_Star.find_route(sprite.pos, target.pos)
    sprite.route = route


class Player(pg.sprite.Sprite):
    player_points = 0
    def __init__(self, game, x, y):
        self._layer = PLAYER_LAYER
        self.groups = game.all_sprites, game.players, game.movers
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.color = 'Blue'
        self.image_file = game.tank_images[self.color]
        self.image = self.image_file.copy()
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.hit_rect = PLAYER_HIT_RECTANGLE.copy()
        self.hit_rect.center = self.rect.center
        self.vel = vec(0, 0)
        self.pos = vec(x, y) * TILESIZE + vec(TILESIZE /2, TILESIZE / 2)
        self.rot = BLUE_PLAYER_INITIAL_ROTATION
        self.last_shot = 0
        self.last_mine = 0
        self.health = PLAYER_HEALTH
        self.bullets = PLAYER_BULLETS
        self.mines = PLAYER_MINES

    def get_keys(self):
        self.vel = vec(0, 0)
        self.rot_speed = 0
        keys = pg.key.get_pressed()
        if keys[pg.K_LEFT] or keys[pg.K_a]:
            self.rot_speed = PLAYER_ROTATION_SPEED
        if keys[pg.K_RIGHT] or keys[pg.K_d]:
            self.rot_speed = -PLAYER_ROTATION_SPEED
        if keys[pg.K_UP] or keys[pg.K_w]:
             self.vel = vec(PLAYER_SPEED, 0).rotate(-self.rot)
        if keys[pg.K_DOWN] or keys[pg.K_s]:
            self.vel = vec(-PLAYER_SPEED / 2, 0).rotate(-self.rot)
        if keys[pg.K_SPACE]:
            shoot_bullet(self)
        if keys[pg.K_RETURN]:
            lay_mine(self)

    def move(self):
        if not self.collide_with_walls():
            self.x += self.dx
            self.y += self.dy

    def update(self):
        self.get_keys()
        self.rot = (self.rot + self.rot_speed * self.game.dt) % 360
        self.image = pg.transform.rotate(self.image_file, self.rot)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.pos += self.vel * self.game.dt
        self.hit_rect.centerx = self.pos.x
        collide_with_walls(self, self.game.walls, 'x')
        self.hit_rect.centery = self.pos.y
        collide_with_walls(self, self.game.walls, 'y')
        self.rect.center = self.hit_rect.center
        if self.health <= 0:
            Explosion(self)
            self.kill()

    def draw_health(self):
        draw_health_box(self, PLAYER_HEALTH)

    # def draw_bullet_counter(self):
    #     draw_bullet_number(self, PLAYER_BULLETS)

    # def draw_mine_counter(self):
    #     draw_mine_number(self, PLAYER_MINES)


class Bullet(pg.sprite.Sprite):
    def __init__(self, sprite, pos, dir):
        self._layer = BULLET_LAYER
        if isinstance(sprite, Mob):
            self.groups = sprite.game.all_sprites, sprite.game.mob_bullets
        else:
            self.groups = sprite.game.all_sprites, sprite.game.player_bullets
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = sprite.game
        self.image = pg.transform.rotate(self.game.bullet_images[sprite.color], sprite.rot)
        self.rect = self.image.get_rect()
        self.pos = vec(pos)
        self.rect.center = pos
        self.vel = dir * BULLET_SPEED
        self.spawn_time = pg.time.get_ticks()

    def update(self):
        self.pos += self.vel * self.game.dt
        self.rect.center = self.pos
        if pg.sprite.spritecollideany(self, self.game.walls):
            self.kill()
        if pg.time.get_ticks() - self.spawn_time > BULLET_LIFETIME:
            self.kill()


class Explosion(pg.sprite.Sprite):
    def __init__(self, sprite):
        self._layer = EFFECT_LAYER
        self.groups = sprite.game.all_sprites, sprite.game.explosion
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = sprite.game
        if sprite in sprite.game.players or sprite.game.mobs:
            self.image = sprite.game.other_images['explosion']
        if sprite in sprite.game.player_mines or sprite.game.player_mines:
            self.image = sprite.game.other_images['mine_explosion']
        self.rect = self.image.get_rect()
        self.pos = vec(sprite.pos)
        self.rect.center = self.pos
        self.spawn_time = now = pg.time.get_ticks()

    def update(self):
        elapsed_time = pg.time.get_ticks() - self.spawn_time
        img_alpha = int((1 - (elapsed_time / EXPLOSION_TIME)) * 255)
        self.image.set_alpha(img_alpha)
        if elapsed_time > EXPLOSION_TIME:
            self.kill()


class Mob(pg.sprite.Sprite):
    mob_counter = 0
    mod_points = 0
    def __init__(self, game, x, y):
        self._layer = MOB_LAYER
        self.groups = game.all_sprites, game.mobs, game.movers
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.mob_ID = Mob.mob_counter
        self.color = 'Red'
        Mob.mob_counter += 1
        self.image_file = game.tank_images[self.color]
        self.image = self.image_file.copy()
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.hit_rect = MOB_HIT_RECTANGLE.copy()
        self.hit_rect.center = self.rect.center
        self.pos = vec(x, y) * TILESIZE + vec(TILESIZE / 2, TILESIZE / 2)
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.rect.center = self.pos
        self.rot = RED_PLAYER_INITIAL_ROTATION
        self.last_shot = 0
        self.last_mine = 0
        self.health = MOB_HEALTH
        self.bullets = MOB_BULLETS
        self.mines = MOB_MINES
        self.A_Star = A_Star(self.game.map_data)
        self.route = [tuple(self.pos)]
        self.next_waypt = tuple(self.pos)
        self.update_time = 0
        self.dis_to_waypt = TILESIZE
        self.moving = False

    def shoot_at_sprite(self, sprite_target): 
        line_of_sight = True
        org_rot = self.rot
        self.rot = (self.game.player.pos - self.pos).angle_to(vec(1, 0))
        for wall in self.game.walls:
            cropped_line = wall.rect.clipline(self.pos, sprite_target.pos)
            if len(cropped_line) != 0:
                line_of_sight = False
                break
        if line_of_sight:
            angle_to_sprite = (sprite_target.pos - self.pos).angle_to(vec(1, 0))
            dis_to_sprite = self.pos.distance_to(sprite_target.pos)
            if abs(angle_to_sprite - self.rot) <= SHOOT_CONE: 
                self.rot = (self.game.player.pos - self.pos).angle_to(vec(1, 0))
                shoot_bullet(self)
        else:
            self.rot = org_rot

    def lay_mine(self): 
        do_mine_rand_draw = random.random()
        if do_mine_rand_draw < MOB_MINE_PROB:
            lay_mine(self)

    def update(self):
        self.get_route()
        self.get_next_waypoint()  
        if self.moving:
            self.rot = (self.next_waypt - self.pos).angle_to(vec(1, 0))
            self.pos += vec(MOB_SPEED, 0).rotate(-self.rot) * self.game.dt
        self.image = pg.transform.rotate(self.image_file, self.rot)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.hit_rect.centerx = self.pos.x
        collide_with_walls(self, self.game.walls, 'x')
        self.hit_rect.centery = self.pos.y
        collide_with_walls(self, self.game.walls, 'y')

        all_mob_entities = [x for x in self.game.mobs.sprites()]
        other_mob_entities = [x for x in all_mob_entities if x.mob_ID != self.mob_ID]
        collide_with_walls(self, other_mob_entities, 'x')
        collide_with_walls(self, other_mob_entities, 'y')

        self.rect.center = self.hit_rect.center
        if self.health <= 0:
            Explosion(self)
            self.kill()

        self.shoot_at_sprite(self.game.player)
        self.lay_mine()

    def draw_health(self):
        draw_health_box(self, MOB_HEALTH)

    # def draw_bullet_counter(self):
    #     draw_bullet_number(self, MOB_BULLETS)

    # def draw_mine_counter(self):
    #     draw_mine_number(self, MOB_MINES)

    def get_route(self):
        self.update_time += self.game.dt
        if self.update_time > MOB_UPDATE_DELAY or len(self.route) <= 1:  
            if self.health <= 2:
                 route_to_closets_health(self)
                 self.moving = True
            elif self.bullets == 0:
                route_to_closets_ammo(self)
                self.moving = True
            else:
                dist_to_player = self.pos.distance_to(self.game.player.pos)
                dist_to_goal = self.pos.distance_to(self.game.goal.pos)
                aggressive_role = random.random()
            
                if dist_to_goal < dist_to_player and aggressive_role > MOB_AGGRESSIVENESS:
                    route_to_target_sprite(self, self.game.goal)
                else:
                    route_to_target_sprite(self, self.game.player)

                if len(self.route) > 1:
                    self.moving = True
                self.update_time = pg.time.get_ticks()

    def get_next_waypoint(self):
        if not hasattr(self, 'next_waypt'):
            self.next_waypt = self.route.pop(0)
        else:
            self.dis_to_waypt = self.pos.distance_to(self.next_waypt)
            if self.dis_to_waypt <= TILESIZE / 10:
                if len(self.route) > 1:
                    self.next_waypt = self.route.pop(0)
                else:
                    self.moving = False


class Wall(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self._layer = WALL_LAYER
        self.groups = game.all_sprites, game.walls
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = pg.Surface((TILESIZE, TILESIZE))
        self.image.fill(GREEN)
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.x = x
        self.y = y
        self.rect.x = x * TILESIZE
        self.rect.y = y * TILESIZE


class Goal(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self._layer = ITEM_LAYER
        self.groups = game.all_sprites, game.goals
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.pos = (vec(x, y) * TILESIZE) + vec(TILESIZE / 2, TILESIZE / 2)
        self.image_file = game.other_images['goal']
        #self.image = pg.transform.scale_by(self.image_file.copy(), 0.5)
        self.image = pg.transform.scale(self.image_file.copy(), (0.75 * TILESIZE, 0.75 * TILESIZE))
        self.rect = self.image.get_rect()
        self.rect.center = self.pos

        
class Ammo(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self._layer = ITEM_LAYER
        self.groups = game.all_sprites, game.ammo_boxes
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.pos = (vec(x, y) * TILESIZE) + vec(TILESIZE / 2, TILESIZE / 2)
        self.image_file = game.other_images['ammo']
        #self.image = pg.transform.scale_by(self.image_file.copy(), 0.5)
        self.image = pg.transform.scale(self.image_file.copy(), (TILESIZE / 2, TILESIZE / 2))
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.available = True
        self.hit_time = 0.

    def update(self):
        if self.available == False:
            if pg.time.get_ticks() - self.hit_time > AMMO_RESPAWN_TIME:
                self.available = True
                self.image.set_alpha(255)


class Health(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self._layer = ITEM_LAYER
        self.groups = game.all_sprites, game.health_kits
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.pos = (vec(x, y) * TILESIZE) + vec(TILESIZE / 2, TILESIZE / 2)
        self.image_file = game.other_images['health']
        #self.image = pg.transform.scale_by(self.image_file.copy(), 0.5)
        self.image = pg.transform.scale(self.image_file.copy(), (TILESIZE / 2, TILESIZE / 2))
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.available = True
        self.hit_time = 0.

    def update(self):
        if self.available == False:
            if pg.time.get_ticks() - self.hit_time > HEALTH_RESPAWN_TIME:
                self.available = True
                self.image.set_alpha(255)
            

class MuzzleFlash(pg.sprite.Sprite):
    def __init__(self, sprite, pos):
        self._layer = EFFECT_LAYER
        self.groups = sprite.game.all_sprites
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = sprite.game
        #size = randint(20, 50)
        scaled_image = pg.transform.scale(self.game.other_images['muzzle_flash'], MUZZLE_FLASH_SIZE)
        self.image = pg.transform.rotate(scaled_image, sprite.rot)
        self.rect = self.image.get_rect()
        self.pos = pos
        self.vel = sprite.vel
        self.rect.center = pos
        self.spawn_time = pg.time.get_ticks()

    def update(self):
        if pg.time.get_ticks() - self.spawn_time > FLASH_DURATION:
            self.kill()
        else:
            self.pos += self.vel * self.game.dt


class Mine(pg.sprite.Sprite):
    def __init__(self, sprite):
        self._layer = BULLET_LAYER
        if isinstance(sprite, Mob):
            self.groups = sprite.game.all_sprites, sprite.game.mob_mines
            self.color = RED
        else:
            self.groups = sprite.game.all_sprites, sprite.game.player_mines
            self.color = BLUE
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = sprite.game
        image = pg.Surface((5, 5))
        image.fill(self.color).copy()
        self.image = pg.transform.rotate(image, 45)
        self.pos = sprite.pos.copy()
        self.rect = self.image.get_rect()
        self.rect.center = sprite.pos
        self.spawn_time = pg.time.get_ticks()

    def update(self):
        if pg.time.get_ticks() - self.spawn_time > MINE_LIFETIME:
            self.kill()

