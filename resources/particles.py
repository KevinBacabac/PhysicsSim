STATIC = False
print('STATIC:', STATIC)


class Particles:
    SUBFRAME = 10

    def __init__(self):
        self.particles = []

    def simulate(self, HEADLESS):
        if STATIC:
            return

        for k in range(self.SUBFRAME):
            for i in self.particles:
                for j in self.particles:
                    if i == j:
                        continue
                    i.calcSpeed(self.SUBFRAME, j)

                i.move(self.SUBFRAME)

                # Disabled when using SDL
                """
                if not HEADLESS:
                    i.draw(windowSurface)
                """

    def save(self):
        export = []
        for i in self.particles:
            export.append(i.export())

        return export
