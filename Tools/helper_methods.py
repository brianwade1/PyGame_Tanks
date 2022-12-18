import pygame as pg
from Config.settings import *

def collide_hit_rect(one, two):
    return one.hit_rect.colliderect(two.rect)

def draw_text(surf, text, color, size, x, y, rotation):
    font = pg.font.Font(FONT_NAME, size)
    text_surface = font.render(text, True, color)
    text_surface = pg.transform.rotate(text_surface, rotation)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surf.blit(text_surface, text_rect)