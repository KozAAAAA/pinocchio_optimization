import matplotlib.pyplot as plt


class Visualizer:
    def __init__(self, space, circles):
        self._space = space
        self._circles = circles

    def draw(self, points):
        fig, ax = plt.subplots()
        ax.set_xlim(self._space["x"])
        ax.set_ylim(self._space["y"])

        for circle in self._circles:
            ax.add_artist(
                plt.Circle((circle["x"], circle["y"]), circle["r"], color="black")
            )

        n_segments = len(points) - 1
        cmap = plt.get_cmap("coolwarm")
        colors = [cmap(i / n_segments) for i in range(n_segments)]

        for i in range(n_segments):
            ax.plot(
                [points[i, 0], points[i + 1, 0]],
                [points[i, 1], points[i + 1, 1]],
                color=colors[i],
                linewidth=2,
            )
            ax.scatter(points[i, 0], points[i, 1], color=colors[i], zorder=3)

        plt.show()
