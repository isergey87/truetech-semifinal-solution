import matplotlib.animation as animation
import matplotlib.pyplot as plt

from constants import MAP_ORIGIN, MAP_RESOLUTION


class Visualizer:
    def __init__(self, grid):
        self.grid = grid
        self.fig, self.ax = plt.subplots()
        # vmin=-1 too dark
        self.imshow = self.ax.imshow(grid.T, origin="lower", cmap="gray_r", vmin=-5, vmax=5)
        self.ax.legend()
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=500, cache_frame_data=False)
        plt.show(block=False)
        plt.pause(0.01)

    def update(self, frame):
        self.imshow.set_data(self.grid.T)
        return self.imshow,

    def render(self):
        plt.pause(0.01)
