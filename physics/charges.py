import math
import pygame

from physics.constants import *

# Rest mass in atomic mass units
ELECTON_MASS = 5.4857990907 * 10**-4
PROTON_MASS = 1.007276466879

# Options
BOUND_WALL = False


def dist(obj1, obj2):
    x1, y1 = obj1
    x2, y2 = obj2

    return math.sqrt((y2-y1)**2 + (x2-x1)**2) * NANO


class ChargedParticle:
    SIZE = 3

    def __init__(self, x, y, dx=0, dy=0):
        self.pos = np.array((x, y), dtype=np.float64)
        self.v = np.array((dx, dy), dtype=np.float64)

    def draw(self, window):
        """ Draws particle when using pygame """
        if self.q > 0:
            colour = pygame.Color('Blue')
        else:
            colour = pygame.Color('Red')

        pygame.draw.circle(window, colour, self.pos, self.SIZE)

    def move(self, subFrame):
        # Nanoseconds * 10^-9
        dTime = 10 * NANO / subFrame
        self.pos += self.v * dTime

        if BOUND_WALL:
            for i in range(len(WIN_DIMS)):
                if self.pos[i] < 0 or self.pos[i] > WIN_DIMS[i] - 1:
                    self.v[i] = 0

                if self.pos[i] < 0:
                    self.pos[i] = 0

                elif self.pos[i] > WIN_DIMS[i] - 1:
                    self.pos[i] = WIN_DIMS[i] - 1

    def calcSpeed(self, subFrame, other):
        distance = dist(self.pos, other.pos)

        if distance > 0:
            # Scalar
            # Positive if same charge
            forceMagnitude = K * self.q * other.q / math.pow(distance, 2)

            # Direction
            # Points out from other object to self
            forceDirection = self.getUnitVector(other.pos)

            # Vector
            acceleration = forceMagnitude * forceDirection / self.m / subFrame
            self.v += acceleration * 5 * NANO**2 + 1

    def export(self):
        return {'x': round(self.pos[0]),
                'y': round(self.pos[1]),
                'q': self.q,
                'v': self.v}

    def getUnitVector(self, pos):
        return -(pos - self.pos) / (np.linalg.norm(pos - self.pos))

    def __repr__(self):
        return str(self.export())


class Lepton:
    m = ELECTON_MASS


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


def initialize(numpy):
    global np
    np = numpy
