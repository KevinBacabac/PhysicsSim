from resources.colour_lib import AngleToColour


def initialize(np):
    global ANGLE

    ANGLE = []
    for i in range(512*3):
        ANGLE.append(AngleToColour(i*360/(512*3)))
    ANGLE = np.array(ANGLE, dtype=np.uint8)
