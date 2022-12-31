# Tank Game

This repo is a top-down tank shooter game designed with PyGame. Players can play against enemy tanks who possess basic decision making ability. The next phase will focus on using reinforcement learning to give the enemy tanks better decision making abilities.

![Main_Scree](/Images/README_images/Main_Screen.PNG)

---

## Environment and Packages

This repo was built with python version 3.10.8. The only package outside the standard packages required for this repo are:

* pygame version 2.1.2
* numpy version 1.22. (needed for the A* algorithm)
* gym version 0.21.0

The [requirements.txt](requirements.txt) allows the user to create an environment with this package by running the command: python3 -m pip install -r requirements.txt

---

## Game Play

The goal of the game is to get more points than your opponent without running out of life within the time limit. Each game currently runs for 2 minutes with a count down timer shown in the upper right of the screen. The current score for red and blue is shown in the upper left.

### Control

Players are the blue tank and the computer controls the red tank. To control the blue tank, use the following commands:

* Left arrow or "a" key = Rotate left
* Right arrow or "d" key = Rotate right
* Up arrow or "w" key = Move ahead
* Down arrow or "s" key = Move backward (backward speed is half the forward speed)
* Space bar = Shoot a bullet
* Return = Lay a mine

The player and computer player start with 10 bullets and 5 mines. Each bullet or mine hit remove 1 point of life.  Mines last for 5 seconds once they are placed down. After the elapsed 5 seconds they disappear from play. Mines are represented by small squares which are the same color as the tank with placed them. The current bullet and mine inventory for both players is shown at the bottom of the screen. These settings can be changed in the [settings.py](Config/settings.py) file.

### Points

Players or the cpu player gains 1 point for each bullet or mine hit. Players also gain 3 points from touching the "goal": ![Goal](/Images/GameImages/explosion1.png =15x15). Once the goal is touched by a tank, it will move to a random open location on the board that is at least 100 units away from any tank.

### Replenish Ammo or Life

During game play, players or the cpu player can regain 3 points of life by touching the "medical kits": ![Med_Kit](/Images/GameImages/HealthPack.png =15x15). Players can also replenish 5 bullets by touching the ![Ammo_Pack](/Images/GameImages/cratewood.png =15x15). Once the health or ammo pack are touched they will disappear and be unavailable for 3 seconds for health or 5 seconds for ammo. These settings can be changed in the [settings.py](Config/settings.py) file

---

## Folders and Files

This repo contains the following folders and files:

* [tank_game](tank_game.py) : Run this file to play the game. This controls the pygame game flow.

* [tank_gym](tank_gym.py) : Creates a Gym environment for training a RL agent to play the [tank_game](tank_game.py).

* [Tools](Tools): This folder contains several classes and methods used in the main script.
  * [sprites.py](Tools/sprites.py) has classes for all the sprites used in the game such as the player tank, cpu-controlled red tank, goal, health kit, etc.
  * [data_loader.py](Tools/data_loader.py) has methods to load the map, images, and settings.
  * [helper_methods.py](Tools/helper_methods.py) has methods used in multiple scripts.
  * [A_Star.py](Tools/A_Star.py) implements the A* algorithm to find the shortest path between two points. This is used by the cpu-controlled red tank to find the shortest path to the player tank, health kits, or ammo kits.

* [Images](Images) : Images used in this Readme file
  * [GameImages](Images/GameImages/): Has the png files used for all images in the game. This is a subset of the full kenney image pack below.
  * [Kenney_topdowntankredux](Images/kenney_topdowntanksredux/) Has the full set of original images from the [kenney_website](https://www.kenney.nl/assets).

* [Config](Config) - Folder for the map files and settings.
  * [map.txt](Config/map.txtmap.txt) map file which is loaded before play. Each "1" is a wall. Each "." is an open square of size (24x24) (size set in the [settings.py](settings.py) file.). The player starts at the "P" location and the red tank(s) start at the "M" locations. Health packs are located at "H" and ammo at "A".
  * [map_complex.txt](Config/map_complex.txt) is a large map which can be used. To use this map, change the setting in the [settings.py](settings.py) file.
  * [settings.py](settings.py) is a python script with all preset settings. This is loaded by all scripts in the repo for reference.

* [requirements.py](requirements.py) - list of python packages required to run the scripts in this repo.

---

## References

Initial code based on examples from: [kids_can_code](https://github.com/kidscancode/pygame_tutorials/tree/master/tilemap)

A star code from: Andrew Jones. "Applying the A* Path Finding Algorithm in Python (Part 1: 2D square grid)", 14 Sept 2018, accessed 20 Dec 2022.
[link](https://www.analytics-link.com/post/2018/09/14/applying-the-a-path-finding-algorithm-in-python-part-1-2d-square-grid)

Game images from : [kenney_topdown-tanks-redux](https://kenney.nl/assets/topdown-tanks-redux)
