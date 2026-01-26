from utils import *


class TargetEncoder(ABC):
    """Abstract base class for muscle spindles"""

    def __init__(self, size, x_bounds=(0, 1), y_bounds=(0, 1)):
        self.size = size
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    @abstractmethod
    def encode(self, x, y):
        pass


class GridTargetEncoder(TargetEncoder):
    def __init__(self, grid_size=10, x_bounds=(0, 1), y_bounds=(0, 1), sigma=0.1):
        super().__init__(
            size=grid_size * grid_size,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        self.grid_size = grid_size
        self.sigma = sigma
        self._initialize_grid()

    def _initialize_grid(self):
        x_centers = np.linspace(self.x_bounds[0], self.x_bounds[1], self.grid_size)
        y_centers = np.linspace(self.y_bounds[0], self.y_bounds[1], self.grid_size)
        self.x_grid, self.y_grid = np.meshgrid(x_centers, y_centers)

    def encode(self, x, y):
        dist_sq = (self.x_grid - x) ** 2 + (self.y_grid - y) ** 2
        return np.exp(-dist_sq / (2 * self.sigma**2))

    def visualize(self, x, y, figsize=(12, 5)):
        h = self.encode(x, y)
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        im = axes[0].imshow(
            h,
            origin="lower",
            cmap="hot",
            extent=[
                self.x_bounds[0],
                self.x_bounds[1],
                self.y_bounds[0],
                self.y_bounds[1],
            ],
        )
        axes[0].plot(x, y, "c*", markersize=15, label="Target")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].set_title("Firing Rate Heatmap")
        axes[0].legend()
        plt.colorbar(im, ax=axes[0], label="Firing Rate")
        ax = fig.add_subplot(122, projection="3d")
        ax.plot_surface(self.x_grid, self.y_grid, h, cmap="hot", alpha=0.8)
        ax.scatter([x], [y], [h.max()], c="cyan", s=100, marker="*", label="Target")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Firing Rate")
        ax.set_title("3D Firing Rate Surface")
        plt.tight_layout()
        plt.show()
        return
