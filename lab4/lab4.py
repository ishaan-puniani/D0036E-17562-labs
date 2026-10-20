"""
Lab 4 — Image processing with scikit-image.

Run from the lab4 folder:
    python lab4.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import data, io
from skimage.color import rgb2gray
from skimage.feature import match_template
from skimage.transform import rescale, resize

# Folder that contains this file (and the two jpg images)
BASE_DIR = Path(__file__).resolve().parent
COINS_PATH = BASE_DIR / "coins.jpg"
ASTRONAUT_PATH = BASE_DIR / "astronaut.jpg"


def to_grayscale(image):
    """Convert RGB to grayscale. Leave a gray image as-is."""
    if image.ndim == 3:
        return rgb2gray(image)
    return image


def print_image_info(name, image):
    """Print shape, dtype, and one pixel value."""
    print(f"\n--- {name} ---")
    print("shape:", image.shape)
    print("dtype:", image.dtype)

    # Color images use [row, col, channel]. Gray images only have [row, col].
    if image.ndim == 3:
        print("pixel [1, 100, 1]:", image[1, 100, 1])
    else:
        print("pixel [1, 100]:", image[1, 100])


def show_image(image, title, cmap=None):
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ---------------------------------------------------------------------------
# Task 1 — Load and visualize an image
# ---------------------------------------------------------------------------

def task1():
    print("\n========== TASK 1: Load and visualize ==========")

    coins = io.imread(COINS_PATH)
    astronaut = io.imread(ASTRONAUT_PATH)

    print_image_info("coins.jpg", coins)
    print_image_info("astronaut.jpg", astronaut)

    show_image(coins, "Task 1: coins.jpg")
    show_image(astronaut, "Task 1: astronaut.jpg")

    return coins, astronaut


# ---------------------------------------------------------------------------
# Task 2 — Color to grayscale
# ---------------------------------------------------------------------------

def task2(coins, astronaut):
    print("\n========== TASK 2: Color space conversion ==========")

    coins_gray = to_grayscale(coins)
    astronaut_gray = to_grayscale(astronaut)

    print_image_info("coins BEFORE gray", coins)
    print_image_info("coins AFTER gray", coins_gray)
    print_image_info("astronaut BEFORE gray", astronaut)
    print_image_info("astronaut AFTER gray", astronaut_gray)

    print(
        "\nWhat changed?\n"
        "  - Color image (astronaut): (height, width, 3), uint8 0..255\n"
        "    after rgb2gray -> (height, width), float64 0.0..1.0\n"
        "  - coins.jpg is already grayscale, so shape and dtype stay the same.\n"
        "  - rgb2gray combines R, G, B into one intensity value."
    )

    show_image(coins_gray, "Task 2: coins grayscale", cmap="gray")
    show_image(astronaut_gray, "Task 2: astronaut grayscale", cmap="gray")

    return coins_gray, astronaut_gray


# ---------------------------------------------------------------------------
# Task 3 — Rescale (factor) vs resize (exact size)
# ---------------------------------------------------------------------------

def task3(astronaut_gray):
    print("\n========== TASK 3: Rescale and resize ==========")

    # resize: you pick the exact output height and width
    astronaut_resized = resize(
        astronaut_gray,
        (200, 300),
        anti_aliasing=True,
    )
    print("original shape: ", astronaut_gray.shape)
    print("resized to 200x300:", astronaut_resized.shape)

    show_image(astronaut_resized, "Task 3: resized to 200 x 300", cmap="gray")

    # rescale: multiply both sides by a factor
    for factor in (0.75, 0.5, 0.25):
        small = rescale(astronaut_gray, factor, anti_aliasing=False)
        print(f"rescaled by {factor}: {small.shape}")
        show_image(small, f"Task 3: rescaled by {factor}", cmap="gray")

    print(
        "\nDifference:\n"
        "  rescale(image, 0.5) -> half the height and half the width\n"
        "  resize(image, (h, w)) -> exactly h rows and w columns"
    )


# ---------------------------------------------------------------------------
# Task 4 — Threshold the coins image
# ---------------------------------------------------------------------------

def task4(coins_gray):
    print("\n========== TASK 4: Thresholding ==========")
    print("grayscale dtype:", coins_gray.dtype)
    print("min / max pixel:", coins_gray.min(), "/", coins_gray.max())

    # Histogram of all pixel values
    plt.figure()
    plt.hist(coins_gray.ravel(), bins=256)
    plt.title("Task 4: coins histogram")
    plt.xlabel("intensity")
    plt.ylabel("count")
    plt.show()

    # Pick t from the dtype (the lab asks you to do this).
    # coins.jpg is already gray uint8 (0..255), so use a value like 100.
    # If the image were float64 from rgb2gray, use a value like 0.45.
    if np.issubdtype(coins_gray.dtype, np.integer):
        t = 100
    else:
        t = 0.45

    binary = coins_gray < t  # True = dark background, False = bright coins

    print("threshold t =", t)
    show_image(binary, f"Task 4: coins < {t}", cmap="gray")


# ---------------------------------------------------------------------------
# Task 5 — Built-in template matching
# ---------------------------------------------------------------------------

def task5():
    print("\n========== TASK 5: Template matching (skimage) ==========")

    # Built-in grayscale coins photo (same one used in the lab handout)
    image = data.coins()
    template = image[170:220, 75:130]

    result = match_template(image, template)

    # Brightest pixel in "result" = best match
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]  # unravel gives (row, col); plot uses (x, y) = (col, row)

    print("best match top-left (x, y) =", (x, y))
    print("best NCC score =", result[y, x])

    _plot_match(template, image, result, x, y, "Task 5: skimage match_template")
    return image, template


# ---------------------------------------------------------------------------
# Task 6 — Own NCC template matching
# ---------------------------------------------------------------------------

def ncc_score(patch, template):
    """
    Normalized Cross-Correlation of two same-size patches.

    1. Subtract the mean (ignore overall brightness).
    2. Divide by the norms (scale the score to about [-1, 1]).
    3. 1.0 = identical shape, 0 = no match, -1 = opposite.
    """
    patch = patch - patch.mean()
    template = template - template.mean()

    numerator = np.sum(patch * template)
    denominator = np.sqrt(np.sum(patch ** 2) * np.sum(template ** 2))

    if denominator == 0:
        return 0.0
    return numerator / denominator


def my_match_template(image, template):
    """
    Slide the template over every valid top-left position in the image.
    Store the NCC score at each position.
    """
    img_h, img_w = image.shape
    tmpl_h, tmpl_w = template.shape

    out_h = img_h - tmpl_h + 1
    out_w = img_w - tmpl_w + 1
    scores = np.zeros((out_h, out_w))

    for row in range(out_h):
        for col in range(out_w):
            patch = image[row : row + tmpl_h, col : col + tmpl_w]
            scores[row, col] = ncc_score(patch, template)

    return scores


def task6(image, template):
    print("\n========== TASK 6: Own template matching (NCC) ==========")
    print("This loops over the image, so it is slower than Task 5. Please wait...")

    result = my_match_template(image, template)

    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    print("best match top-left (x, y) =", (x, y))
    print("best NCC score =", result[y, x])
    print(
        "This should be the same coin as Task 5.\n"
        "skimage uses the same NCC idea, just computed much faster."
    )

    _plot_match(template, image, result, x, y, "Task 6: our NCC match")


def _plot_match(template, image, result, x, y, title):
    h, w = template.shape

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    fig.suptitle(title)

    axes[0].imshow(template, cmap="gray")
    axes[0].set_title("template")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")
    axes[1].set_title("image")
    axes[1].axis("off")
    axes[1].add_patch(
        plt.Rectangle((x, y), w, h, edgecolor="r", facecolor="none")
    )

    axes[2].imshow(result)
    axes[2].set_title("NCC scores")
    axes[2].axis("off")
    axes[2].plot(x, y, "o", markeredgecolor="r", markerfacecolor="none", markersize=10)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Run all tasks
# ---------------------------------------------------------------------------

def main():
    coins, astronaut = task1()
    coins_gray, astronaut_gray = task2(coins, astronaut)
    task3(astronaut_gray)
    task4(coins_gray)
    image, template = task5()
    task6(image, template)


if __name__ == "__main__":
    main()
