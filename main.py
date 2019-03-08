import colourLib #Custom library
import numpy as np
import random
import math
import shelve  # For exporting data
import time
import pygame

#Optional replacement for numpy which uses gpu computing
#import bohrium as np

try:
    # Faster numpy execution
    import numexpr as ne
    raise ImportError
except ImportError:
    ne = None

#Constants of nature
TAU = math.pi * 2

#Electron rest mass in kilograms
ME = 9.10938291 * 10**(-31)
NANO = 10**(-9) #Nanometers or nanoseconds to meters or seconds

E = 1.602 * 10**(-19)
K = 8.99 * 10**9 / (NANO**2) #N * nm^2 * C^-2
PROTON_MASS = 1.6726 * 10**(-27)

#Options
WIN_DIMS = (700, 700)

subFrame = 2000
HEADLESS = False
BOUND_WALL = True
DUMP_DATA = False
CALC_FIELD = True
STATIC = False

print('STATIC:', STATIC)

if HEADLESS:
    windowSurface = pygame.Surface(WIN_DIMS)

else:
    windowSurface = pygame.display.set_mode(WIN_DIMS)
    mainClock = pygame.time.Clock()
    FPS = 30

pygame.init()

distanceSq = math.pow(100*NANO, 2)
maxElectric = math.sqrt(K * E / distanceSq)
NANO_FACTOR = 255 / maxElectric

#Superimposes b2 on b1
def addArrays(b1, b2, xy):
    assert len(xy) == 2

    x, y = round(xy[0]), round(xy[1])
    W_WIDTH, W_HEIGHT = WIN_DIMS

    minX = int(x - W_WIDTH)
    maxX = int(x + W_WIDTH)
    minY = int(y - W_HEIGHT)
    maxY = int(y + W_HEIGHT)
    #Block was from elsewhere
    #Check if any part is on screen, would crash otherwise
    condition = not (maxX < b1.shape[0] or  #Rightmost is to the left of right window
                     minX >= 0 or           #The left is right of the left of window
                     maxY < b1.shape[1] or  #The bottom is above the bottom of window
                     minY >= 0)             #Top is below the top of window


    if condition:
        v_range1 = slice(max(0, minX), max(min(minX + b2.shape[0], b1.shape[0]), 0))
        h_range1 = slice(max(0, minY), max(min(minY + b2.shape[1], b1.shape[1]), 0))

        v_range2 = slice(max(0, -minX), min(-minX + b1.shape[0], b2.shape[0]))
        h_range2 = slice(max(0, -minY), min(-minY + b1.shape[1], b2.shape[1]))

        b1[v_range1, h_range1] += b2[v_range2, h_range2]

    return b1

ANGLE_LOOKUP = []
for i in range(512*3):
    ANGLE_LOOKUP.append(colourLib.AngleToColour(i*360/(512*3)))
ANGLE_LOOKUP = np.array(ANGLE_LOOKUP, dtype=np.uint8)

def renderArray(electricField):
    """Return surface as numpy array with electric field visualized by colour"""
    def makeDrawnField(electricField):
        #Check if valid
        shape = electricField.shape
        #3 dimensional, with 2 dimensions as x,y coords and third dimension for x/y components
        assert len(shape) == 3
        assert shape[0:2] == WIN_DIMS

        QUARTER_ROT = TAU / 4
        MAX_INDICES = 3 * 512

        #Target surface
        display = np.zeros((*WIN_DIMS, 4), dtype=np.uint8)

        #Build a 2d array of norms using the x/y components (3rd dimension of array)
        Forces = np.linalg.norm(electricField, axis=2)

        #Separate components
        y = electricField[:,:,1]
        x = electricField[:,:,0]

        #Calculate the hue angles
        if ne:
            angleIndices = ne.evaluate('(arctan2(y, x) + QUARTER_ROT) * MAX_INDICES / TAU')

        else:
            Rotations = np.arctan2(y, x)
            Rotations += QUARTER_ROT

            angleIndices = Rotations * MAX_INDICES / TAU

        #Floor operation returns floats
        angleIndices = np.floor(angleIndices).astype(int)
        angleIndices %= MAX_INDICES


        #https://stackoverflow.com/questions/14448763/is-there-a-convenient-way-to-apply-a-lookup-table-to-a-large-array-in-numpy
        #Set RGB using hue angles dictated by direction
        display[:,:,:3] = ANGLE_LOOKUP[angleIndices]

        #Set saturation to be proportional to force
        ForceDisplay = np.sqrt(Forces) * NANO_FACTOR
        ForceDisplay = np.clip(ForceDisplay, 0, 255)
        display[:,:,3] = np.asarray(ForceDisplay, np.uint8)

        return display

    numpyArray = makeDrawnField(electricField)
    background = pygame.surfarray.make_surface(numpyArray[:,:,0:3])
    surface = pygame.Surface(numpyArray.shape[0:2], flags=pygame.SRCALPHA)
    surface.blit(background, (0, 0))
    temp = pygame.surfarray.pixels_alpha(surface)
    temp[:,:] = numpyArray[:,:,3]
    del temp

    return surface

if not HEADLESS:
    surface = pygame.Surface((2001, 2001), flags=pygame.SRCALPHA)

def dist(obj1, obj2):
    x1, y1 = obj1
    x2, y2 = obj2

    return math.sqrt((y2-y1)**2 + (x2-x1)**2) * NANO

class ChargedParticle:
    def __init__(self, x, y, dx=0, dy=0):
        self.pos = np.array((x, y), dtype=np.float64)
        self.v = np.array((dx, dy), dtype=np.float64)

        if not HEADLESS:
            self.rect = pygame.Rect(*self.pos, 1, 1)
            self.rect = self.rect.inflate(4, 4)

    def draw(self):
        if not HEADLESS:
            self.rect = pygame.Rect(*self.pos, 1, 1)
            self.rect = self.rect.inflate(4, 4)

            if self.q > 0:
                colour = pygame.Color('Blue')
            else:
                colour = pygame.Color('Red')

    def move(self):
        #Femtoseconds * 10^-18
        self.pos += self.v * NANO**2 / subFrame

        if BOUND_WALL:
            for i in range(len(WIN_DIMS)):
                if False:
                    if self.pos[i] < 0 or self.pos[i] > WIN_DIMS[i] - 1:
                        self.v[i] = 0

                    if self.pos[i] < 0:
                        self.pos[i] = 0

                    elif self.pos[i] > WIN_DIMS[i] - 1:
                        self.pos[i] = WIN_DIMS[i] - 1

    def calcSpeed(self, other):
        distance = dist(self.pos, other.pos)

        if distance > 0:
            #Scalar
            #Positive if same charge
            forceMagnitude = K * self.q * other.q / math.pow(distance, 2)

            #Direction
            #Points out from other object to self
            forceDirection = -(other.pos - self.pos) / (np.linalg.norm(other.pos - self.pos))

            #Vector
            forceVector = forceMagnitude * forceDirection
            acceleration = forceVector / self.m * NANO**2 / subFrame

            self.v += acceleration

    def export(self):
        return {'x': round(self.pos[0]),
                'y': round(self.pos[1]),
                'q': self.q,
                'v': self.v}

    def __repr__(self):
        return str(self.export())

class Lepton:
    m = 9.1094 * 10**(-31)

class Electron(ChargedParticle, Lepton):
    q = -E

class Positron(ChargedParticle, Lepton):
    q = E

class Proton(ChargedParticle):
    q = E
    m = PROTON_MASS

class AntiProton(ChargedParticle):
    q = -E
    m = PROTON_MASS

particles = []

'''
# Dipole moment
particles.append(Positron(349, 350))
particles.append(Electron(350, 350))
'''

#Quadropole moment
particles.append(Positron(349, 350))
particles.append(Positron(351, 350))
particles.append(Electron(350, 349))
particles.append(Electron(350, 351))

class ElectricField:
    results = {}
    subParts = 1

    @staticmethod
    def get(q, pos):
        offset = pos % 1 #Preserve only decimal

        #Turn decimals into indices
        '''
        >>> a
        array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        >>> np.floor(a * 2)
        array([0., 0., 0., 0., 1., 1., 1., 1., 1.])
        >>> np.floor(a * 4)
        array([0., 0., 1., 1., 2., 2., 2., 3., 3.])
        '''
        offset = tuple(np.floor(offset * ElectricField.subParts))
        key = (q,) + offset #Make 3 element tuple (charge, xOffset, yOffset)

        try:
            return ElectricField.results[key]
        except KeyError:
            ElectricField.results[key] = ElectricField.calculateField(K, NANO, WIN_DIMS[0]*2+1, key)

        return ElectricField.results[key]

    def calculateField(K, NANO, size, key):
        def getDistSqArray(size):
            #https://stackoverflow.com/questions/26033672/creating-a-2-dimensional-numpy-array-with-the-euclidean-distance-from-the-center
            x_inds, y_inds = np.ogrid[:size, :size]

            def getDist(x, y, mid):
                return (((y - mid)*NANO)**2 + ((x - mid)*NANO)**2)

            return getDist(x_inds, y_inds, round(size/2))

        def getAtan2Array(size):
            x_inds, y_inds = np.ogrid[:size, :size]

            def getDeltaY(x, y, mid):
                return y - mid

            def getDeltaX(x, y, mid):
                return x - mid

            deltaY = getDeltaY(x_inds, y_inds, round(size/2))
            deltaX = getDeltaX(x_inds, y_inds, round(size/2))

            return np.arctan2(deltaY, deltaX)

        q, xOffset, yOffset = key

        electricField = np.zeros((size, size, 2))

        mid = round(size/2)

        distanceSq = getDistSqArray(size)

        if distanceSq[mid][mid] == 0:
            distanceSq[mid][mid] = np.finfo(np.float16).max # No force at the point, distance is 0
            print('Processing field - ' + str(key))

        Force = K * q / distanceSq

        Rotations = getAtan2Array(size)
        electricField[:,:,0] = Force * np.cos(Rotations)
        electricField[:,:,1] = Force * np.sin(Rotations)

        return electricField


def exportParticles(particles):
    export = []
    for i in particles:
        export.append(i.export())

    return export


def main():
    logs = []
    fields = []
    screencount = 0
    start = time.time()

    while True:
        electricField = np.zeros((*WIN_DIMS, 2))
        exitGame = False

        #Refresh screen
        windowSurface.fill(pygame.Color('Black'))

        if not HEADLESS:
            #Store events so they can be checked multiple times
            pygameEvents = pygame.event.get()

            for event in pygameEvents:
                if event.type == pygame.QUIT:
                    exitGame = True

        if exitGame:
            pygame.quit()
            break

        #Simulate particles
        if not STATIC:
            for k in range(subFrame // 10):
                for i in particles:
                    for j in particles:
                        if i == j:
                            continue
                        i.calcSpeed(j)


                    i.move()

        for i in particles:
            i.draw()

            if CALC_FIELD:
                electricField = addArrays(electricField, ElectricField.get(i.q, i.pos), i.pos)

        if CALC_FIELD:
            surface = renderArray(electricField)
            windowSurface.blit(surface, (0, 0))

        print(round(1 / (time.time() - start), 2), 'FPS')
        start = time.time()
        if not HEADLESS:
            pygame.display.update()

        if DUMP_DATA:
            logs.append(exportParticles(particles))
            fields.append(electricField.copy())

            saveFile = shelve.open('export')
            saveFile['field'] = fields
            saveFile['particles'] = logs
            saveFile.close()

        screencount += 1

main()
