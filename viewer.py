# viewer.py

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np

def show_projections_keyboard(projections: np.ndarray) -> None:
    """Interactive viewer using keyboard (left/right arrow keys)."""
    num_images = projections.shape[0]
    idx = [0]  # mutable container for current index

    fig, ax = plt.subplots()
    img_display = ax.imshow(projections[idx[0]], cmap='gray')
    ax.set_title(f"Projection {idx[0] + 1}/{num_images}")

    def on_key(event):
        if event.key == 'right':
            idx[0] = (idx[0] + 1) % num_images
        elif event.key == 'left':
            idx[0] = (idx[0] - 1) % num_images
        else:
            return
        img_display.set_data(projections[idx[0]])
        ax.set_title(f"Projection {idx[0] + 1}/{num_images}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.show()


def show_projections_slider(projections: np.ndarray) -> None:
    """Interactive viewer using a slider."""
    num_proj = projections.shape[0]
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)
    img_display = ax.imshow(projections[0], cmap='gray')
    ax.set_title("Projection 0")

    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Index', 0, num_proj - 1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        img_display.set_data(projections[idx])
        ax.set_title(f"Projection {idx}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()
