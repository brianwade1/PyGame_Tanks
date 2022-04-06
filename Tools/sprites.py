import pygame as pg
from Config.settings import *
from Tools.helper_methods import collide_hit_rect

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
    #pg.draw.rect(sprite.image, col, sprite.health_bar)
    if sprite.health < full_health:
        pg.draw.rect(sprite.image, col, sprite.health_bar)

def shoot_bullet(sprit):
    now = pg.time.get_ticks()
    if now - sprit.last_shot > BULLET_RATE_DELAY:
        sprit.last_shot = now
        dir = vec(1, 0).rotate(-sprit.rot)
        pos = sprit.pos + BARREL_OFFSET.rotate(-sprit.rot)
        Bullet(sprit, pos, dir)
        sprit.vel = vec(-KICKBACK, 0).rotate(-sprit.rot)
        MuzzleFlash(sprit, pos)


class Player(pg.sprite.Sprite):
    def __init__(self, game, x, y):
        self._layer = PLAYER_LAYER
        self.groups = game.all_sprites, game.players
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
        self.pos = vec(x, y) * TILESIZE
        self.rot = BLUE_PLAYER_INITIAL_ROTATION
        self.last_shot = 0
        self.health = PLAYER_HEALTH

    def get_keys(self):
        # for event in pg.event.get():
        #     if event.type == pg.KEYDOWN and event.key == pg.K_SPACE:
        #         dir = vec(1, 0).rotate(-self.rot)
        #         pos = self.pos #+ BARREL_OFFSET.rotate(-self.rot)
        #         Bullet(self.game, pos, dir, self.rot, self.color)
        #         self.vel = vec(-KICKBACK, 0).rotate(-self.rot)


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
            Explosion(self.game, self.pos)
            self.kill()

    def draw_health(self):
        draw_health_box(self, PLAYER_HEALTH)


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
    def __init__(self, game, pos):
        self._layer = EFFECT_LAYER
        self.groups = game.all_sprites, game.explosion
        pg.sprite.Sprite.__init__(self, self.groups)
        self.game = game
        self.image = game.other_images['explosion']
        self.rect = self.image.get_rect()
        self.pos = vec(pos)
        self.rect.center = pos
        self.spawn_time = now = pg.time.get_ticks()

    def update(self):
        elapsed_time = pg.time.get_ticks() - self.spawn_time
        img_alpha = int((1 - (elapsed_time / EXPLOSION_TIME)) * 255)
        self.image.set_alpha(img_alpha)
        if elapsed_time > EXPLOSION_TIME:
            self.kill()


class Mob(pg.sprite.Sprite):
    mob_counter = 0
    def __init__(self, game, x, y):
        self._layer = MOB_LAYER
        self.groups = game.all_sprites, game.mobs
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
        self.pos = vec(x, y) * TILESIZE
        self.vel = vec(0, 0)
        self.acc = vec(0, 0)
        self.rect.center = self.pos
        self.rot = RED_PLAYER_INITIAL_ROTATION
        self.last_shot = 0
        self.health = MOB_HEALTH

    def shoot_at_sprite(self, sprite_target): 
        line_of_sight = True
        for wall in self.game.walls:
            cropped_line = wall.rect.clipline(self.pos, sprite_target.pos)
            if len(cropped_line) != 0:
                line_of_sight = False
                break
        if line_of_sight:
            angle_to_sprite = (sprite_target.pos - self.pos).angle_to(vec(1, 0))
            dis_to_sprite = self.pos.distance_to(sprite_target.pos)
            if abs(angle_to_sprite - self.rot) <= SHOOT_CONE: 
                shoot_bullet(self)

    def update(self):
        self.rot = (self.game.player.pos - self.pos).angle_to(vec(1, 0))
        self.image = pg.transform.rotate(self.image_file, self.rot)
        self.rect = self.image.get_rect()
        self.rect.center = self.pos
        self.acc = vec(MOB_SPEED, 0).rotate(-self.rot)
        self.acc += self.vel * -1
        self.vel += self.acc * self.game.dt
        self.pos += self.vel * self.game.dt + 0.5 * self.acc * self.game.dt ** 2
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
            Explosion(self.game, self.pos)
            self.kill()

        self.shoot_at_sprite(self.game.player)

    def draw_health(self):
        draw_health_box(self, MOB_HEALTH)


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