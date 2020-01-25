import random
import math
import shelve  # For exporting data
import time
import pygame
import sdl2.ext

try:
    import bohrium as np  # GPU computing support
except ImportError:
    import numpy as np

try:
    import numexpr as ne
except ImportError:
    ne = None


import resources.lookup as lookup
from resources.particles import Particles

import physics.charges as charges
import physics.efield as efield

charges.initialize(np)
efield.initialize(np)
lookup.initialize(np)

# Constants of nature
TAU = math.pi * 2

# Options
WIN_DIMS = (500, 360)
HEADLESS = False
DUMP_DATA = False
CALC_FIELD = True

if HEADLESS:
    windowSurface = pygame.Surface(WIN_DIMS)

else:
    sdl2.ext.init()
    window = sdl2.ext.Window('Charge Simulation!', size=WIN_DIMS)
    window.show()
    factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)
    renderer = factory.create_sprite_render_system(window)
    field = factory.from_color(sdl2.ext.Color(100, 120, 255), size=WIN_DIMS)

    mainClock = pygame.time.Clock()
    FPS = 30

pygame.init()

E = 1
distanceSq = math.pow(50, 2)
maxElectric = math.sqrt(E / distanceSq)
SATURATION_FACTOR = 255 / maxElectric


def render_array(electric_field, field):
    # Return surface as numpy array with electric field visualized by colour
    global arr

    def make_drawn_field(electric_field):
        # Check if valid
        shape = electric_field.shape
        # 3 dimensional, with 2 dimensions as x,y coords and third dimension for x/y components
        assert len(shape) == 3
        assert shape[0:2] == WIN_DIMS

        QUARTER_ROT = TAU / 4
        MAX_INDICES = 3 * 512

        # Target surface
        display = np.zeros((*WIN_DIMS, 4), dtype=np.uint8)

        # Build a 2d array of norms using the x/y components (3rd dimension of array)
        Forces = np.linalg.norm(electric_field, axis=2)

        # Separate components
        y = electric_field[:, :, 1]
        x = electric_field[:, :, 0]

        # Calculate the hue angles
        if ne:
            angleIndices = ne.evaluate('(arctan2(y, x) + QUARTER_ROT) * MAX_INDICES / TAU')

        else:
            Rotations = np.arctan2(y, x)
            Rotations += QUARTER_ROT

            angleIndices = Rotations * MAX_INDICES / TAU

        # Floor operation returns floats
        angleIndices = np.floor(angleIndices).astype(int)
        angleIndices %= MAX_INDICES

        # https://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
        # Set RGB using hue angles dictated by direction
        display[:, :, :3] = lookup.ANGLE[angleIndices]

        # Set saturation to be proportional to force
        ForceDisplay = np.sqrt(Forces) * SATURATION_FACTOR / 255  # 0 to 1 alpha
        ForceDisplay = np.clip(ForceDisplay, 0, 1)
        display[:, :, 0] = np.multiply(display[:, :, 0], ForceDisplay)
        display[:, :, 1] = np.multiply(display[:, :, 1], ForceDisplay)
        display[:, :, 2] = np.multiply(display[:, :, 2], ForceDisplay)

        return display

    numpyArray = make_drawn_field(electric_field)
    arr = sdl2.ext.pixels3d(field)
    arr[:, :] = numpyArray[:, :]

    del arr

    return field


particles = Particles()
particles.particles.append(charges.Positron(100, 50))
particles.particles.append(charges.Electron(400, 50))

v = 200000000
particles.particles[0].v[1] = -v
particles.particles[1].v[1] = v


def main():
    logs = []
    fields = []
    frame_count = 0
    start = time.time()

    while True:
        exitGame = False

        if not HEADLESS:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    exitGame = True

            events = sdl2.ext.get_events()
            for event in events:
                if event.type == sdl2.SDL_QUIT:
                    exitGame = True
                    break

        if exitGame:
            pygame.quit()
            sdl2.ext.quit()
            break

        # Simulate particles
        particles.simulate(HEADLESS)

        if CALC_FIELD:
            electric_field = efield.get_field(WIN_DIMS, particles)

            # Modify
            render_array(electric_field, field)

        #print('Finished render', frame_count, 'after', round(1000 * (time.time() - start)), ' milliseconds.')
        print(round(1 / (time.time() - start), 1), 'FPS')
        start = time.time()
        if not HEADLESS:
            renderer.render(field)
            window.refresh()
            mainClock.tick(FPS)

        if DUMP_DATA:
            logs.append(particles.save())
            fields.append(electric_field.copy())

            saveFile = shelve.open('export')
            saveFile['field'] = fields
            saveFile['particles'] = logs
            saveFile.close()

        #pygame.image.save(windowSurface, str(frame_count) + '.png')
        frame_count += 1


main()
