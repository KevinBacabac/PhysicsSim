from physics.constants import *


# Superimposes b2 on b1
def add_arrays(WIN_DIMS, b1, b2, xy):
    assert len(xy) == 2

    x, y = round(xy[0]), round(xy[1])
    W_WIDTH, W_HEIGHT = WIN_DIMS

    minX = int(x - W_WIDTH)
    maxX = int(x + W_WIDTH)
    minY = int(y - W_HEIGHT)
    maxY = int(y + W_HEIGHT)
    # Block was from elsewhere
    # Check if any part is on screen, would crash otherwise
    condition = not (maxX < b1.shape[0] or  # Rightmost is to the left of right window
                     minX >= 0 or           # The left is right of the left of window
                     maxY < b1.shape[1] or  # The bottom is above the bottom of window
                     minY >= 0)             # Top is below the top of window
    if condition:
        v_range1 = slice(max(0, minX), max(min(minX + b2.shape[0], b1.shape[0]), 0))
        h_range1 = slice(max(0, minY), max(min(minY + b2.shape[1], b1.shape[1]), 0))

        v_range2 = slice(max(0, -minX), min(-minX + b1.shape[0], b2.shape[0]))
        h_range2 = slice(max(0, -minY), min(-minY + b1.shape[1], b2.shape[1]))

        b1[v_range1, h_range1] += b2[v_range2, h_range2]

    return b1


def get_field(WIN_DIMS, particles):
    electricField = np.zeros((*WIN_DIMS, 2))
    for i in particles.particles:
        electricField = add_arrays(WIN_DIMS, electricField,
                                  ElectricField.get(WIN_DIMS, i.q, i.pos), i.pos)

    return electricField


class ElectricField:
    results = {}
    subParts = 1

    @staticmethod
    def get(WIN_DIMS, q, pos):
        offset = pos % 1  # Preserve only decimal

        # Turn decimals into indices
        '''
        >>> a
        array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        >>> np.floor(a * 2)
        array([0., 0., 0., 0., 1., 1., 1., 1., 1.])
        >>> np.floor(a * 4)
        array([0., 0., 1., 1., 2., 2., 2., 3., 3.])
        '''
        offset = tuple(np.floor(offset * ElectricField.subParts))
        key = (q,) + offset  # Make 3 element tuple (charge, xOffset, yOffset)

        try:
            return ElectricField.results[key]
        except KeyError:
            ElectricField.results[key] = ElectricField.calculateField(K, NANO, WIN_DIMS[0]*2+1, key)

        return ElectricField.results[key]

    def calculateField(K, NANO, size, key):
        def getDistSqArray(size):
            # https://stackoverflow.com/questions/26033672/creating-a-2-dimensional-numpy-array-with-the-euclidean-distance-from-the-center
            x_inds, y_inds = np.ogrid[:size, :size]

            def getDist(x, y, mid):
                return ((y - mid)**2 + (x - mid)**2)

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
            distanceSq[mid][mid] = np.finfo(np.float16).max  # No force at the point, distance is 0
            print('Processing field -', key)

        # Proportional to charge
        Force = q / distanceSq

        Rotations = getAtan2Array(size)
        electricField[:, :, 0] = Force * np.cos(Rotations)
        electricField[:, :, 1] = Force * np.sin(Rotations)

        return electricField


def initialize(numpy):
    global np
    np = numpy
