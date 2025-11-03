from os import path
import pygame as pg

# DISPLAY SETTINGS
WIDTH = 1920
HEIGHT = 1080

# GAME SETTINGS
TITLE = "Agar.Pio"
FPS = 60
SPEED_DECAY = 0.9  # Cell speed reduced by X% per second

# PLAYER SETTINGS
MIN_PLAYER_MASS = 10.0
MASS_LOSS_RATE = 0.98  # Cell mass reduced by X% per second
EAT_PROPORTION = 0.9  # Can only eat cells X% your size or smaller
SPLIT_LIMIT = 16  # Max number of cells a player can have
SPLIT_SPEED = 3.0
EJECT_MASS = 13.0
EJECT_SPEED = 1.5
BASE_MERGE_DELAY = 30.0  # Seconds before cells can merge together
MERGE_DELAY_INCREMENT = 0.02333  # Merge delay is extended by X% of your cell's mass

# VIRUS SETTINGS
VIRUS_EAT_MASS = 100.0

#IMAGES 
BACKGROUND = pg.image.load(path.join('images','background.png'))