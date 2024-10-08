import cv2 as cv
import numpy as np
import pyautogui
import typing as tp
from time import time_ns
from enum import Enum


###################################################################
#                  TYPE DEFINITIONS AND CONSTANTS                 #
###################################################################

WINDOW_NAME = 'Minesweeper Cheat'
MAX_FPS = 5
NONE_RECT = ((-1, -1), (-1, -1))
BLACK = (0, 0, 0)
RED = (0, 0, 255)
EMPTY = 0
UNCLICKED = 9
FLAGGED = 10
UNMATCHED = 255


Pos = tp.Tuple[int, int]
Rect = tp.Tuple[Pos, Pos]
ImgGray = np.ndarray
ImgBGR = np.ndarray
ImgHSV = np.ndarray
ImgBool = np.ndarray
HSVColor = tp.Tuple[int, int, int]
Uint8_2D = np.ndarray
Thresholds = tp.Dict[int, tp.Tuple[tp.Tuple[HSVColor, HSVColor], ...]]
RectGrid = tp.List[tp.List[Rect]]
GridState = np.ndarray


# HSV input format: hue in [0, 360], saturation in [0, 100], value in [0, 100]
LIGHT_MODE_STATE_HSV_COLORS = {
    0: (0, 0, 78),
    1: (240, 100, 97),
    2: (120, 100, 47),
    3: (0, 100, 93),
    4: (240, 100, 50),
    5: (0, 100, 50),
    6: (180, 100, 50),
    7: (0, 0, 0),
    8: (0, 0, 44),
    UNCLICKED: (0, 0, 78),
}
DARK_MODE_STATE_HSV_COLORS = {
    0: (210, 22, 28),
    1: (206, 51, 100),
    2: (120, 47, 75),
    3: (352, 53, 100),
    4: (291, 47, 100),
    5: (44, 85, 87),
    6: (180, 50, 80),
    7: (0, 0, 60),
    8: (210, 7, 88),
    UNCLICKED: (210, 18, 36),
}
NIGHT_SHIFT_STATE_HSV_COLORS = {
    0: (0, 0, 20),
    1: (210, 54, 87),
    2: (120, 50, 63),
    3: (350, 50, 80),
    4: (280, 46, 87),
    5: (54, 100, 67),
    6: (180, 50, 67),
    7: (0, 0, 60),
    8: (0, 0, 80),
    UNCLICKED: (0, 0, 27),
}


class ColorChannel:
    RED = 2
    GREEN = 1
    BLUE = 0


class ColorMode(Enum):
    LIGHT = 0
    DARK = 1
    NIGHT_SHIFT = 2
    UNKNOWN = 3


###################################################################
#                  MINESWEEPER STATE RECOGNITION                  #
###################################################################

def colors_to_thresholds(colors: tp.Dict[int, HSVColor], brightness: int,
                         no_thresholds: bool, value_threshold: int) -> Thresholds:
    # Output HSV format: hue in [0, 180], saturation in [0, 255], value in [0, 255]

    HUE_THRESHOLD = 2
    SATURATION_THRESHOLD = 5
    VALUE_THRESHOLD = value_threshold
    if no_thresholds:
        HUE_THRESHOLD = SATURATION_THRESHOLD = VALUE_THRESHOLD = 0

    thresholds: Thresholds = {}
    for key, color in colors.items():
        hue = color[0] // 2
        hue_low: int = (hue - HUE_THRESHOLD) % 180
        hue_high: int = (hue + HUE_THRESHOLD) % 180
        sat = int(color[1] * 2.55)
        sat_low: int = max(0, sat - SATURATION_THRESHOLD)
        sat_high: int = min(255, sat + SATURATION_THRESHOLD)
        val = int(color[2] * 2.55 * brightness / 100)
        val_low: int = max(0, val - VALUE_THRESHOLD)
        val_high: int = min(255, val + VALUE_THRESHOLD)

        if hue_low < hue_high:
            thresholds[key] = (((hue_low, sat_low, val_low), (hue_high, sat_high, val_high)),)
        else:
            thresholds[key] = (((0, sat_low, val_low), (hue_high, sat_high, val_high)),
                                ((hue_low, sat_low, val_low), (180, sat_high, val_high)))

    return thresholds


def state_thresholds(color_mode: ColorMode, brightness: int,
                     no_thresholds: bool = False,
                     value_threshold: int = 10) -> Thresholds:
    """ Get the color thresholds for the states of the tiles in the minefield
        for the given color mode and brightness.

        Args:
            color_mode (ColorMode): The color mode of the game. Can't be ColorMode.UNKNOWN.
            brightness (int): The brightness of the screen. In [50, 100].
            no_thresholds (bool, optional): Whether to use no thresholds. Defaults to False.
            value_threshold (int, optional): The threshold for the value of the colors.
                Defaults to 10.

        Returns:
            Thresholds: The color thresholds for the states.
    """

    if color_mode == ColorMode.LIGHT:
        return colors_to_thresholds(LIGHT_MODE_STATE_HSV_COLORS, brightness,
                                    no_thresholds, value_threshold)
    elif color_mode == ColorMode.DARK:
        return colors_to_thresholds(DARK_MODE_STATE_HSV_COLORS, brightness,
                                    no_thresholds, value_threshold)
    elif color_mode == ColorMode.NIGHT_SHIFT:
        return colors_to_thresholds(NIGHT_SHIFT_STATE_HSV_COLORS, brightness,
                                    no_thresholds, value_threshold)
    else:
        raise ValueError('{} is not supported'.format(color_mode))


def find_minefield(screen: ImgBGR) -> Rect:
    """ Find the minefield in the screen image by detecting the largest rectangle.

        Args:
            screen (ImgBGR): The screen image.

        Returns:
            Rect: The rectangle of the minefield in the form ((left, top), (right, bottom)).
    """

    edges = cv.Canny(screen, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    minefield_area = 0
    best_rect = ((0, 0), (0, 0))
    for cnt in contours:
        left, top, width, height = cv.boundingRect(cnt)
        rect_area = height * width
        if rect_area > minefield_area:
            minefield_area = rect_area
            best_rect = ((left, top), (left + width, left + height))

    return best_rect


def extract_gray(img: ImgBGR, color_mode: ColorMode) -> ImgBGR:
    """ Mask out all non-gray pixels in the image. Gray pixels are defined as pixels
        that have the same value in all three color channels, or pixels that have their blue
        channel value slightly higher than the green channel value and the red channel value
        slightly lower than the green channel value, with these differences not higher than
        DARK_MODE_THRESHOLD (corresponds to minesweeper.online dark mode tile colors).

        Args:
            bgr_img (ImgBGR): The image to mask. Has to be in BGR format.
            color_mode (ColorMode): The color mode of the image.

        Returns:
            ImgBGR: The masked image.
    """

    BLACK_THRESHOLD = 15
    WHITE_THRESHOLD = 240
    DARK_MODE_THRESHOLD = 10 #? Same as in detect_color_mode

    if color_mode == ColorMode.LIGHT:
        WHITE_THRESHOLD = 255
    elif color_mode in (ColorMode.DARK, ColorMode.NIGHT_SHIFT):
        BLACK_THRESHOLD = 1

    # Handle exactly gray pixels (not blueish)
    #? Same as in detect_color_mode
    mask = (img[:, :, 0] == img[:, :, 1]) & (img[:, :, 1] == img[:, :, 2])

    # Handle blueish (dark mode) pixels
    #? Same as in detect_color_mode
    mask |= (((img[:, :, ColorChannel.BLUE] - img[:, :, ColorChannel.GREEN])
              <= DARK_MODE_THRESHOLD)
             & ((img[:, :, ColorChannel.GREEN] - img[:, :, ColorChannel.RED])
                <= DARK_MODE_THRESHOLD))

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask &= (gray_img >= BLACK_THRESHOLD) & (gray_img <= WHITE_THRESHOLD)

    img = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))
    return img


def detect_color_mode(img: ImgBGR) -> ColorMode:
    """ Detect the color mode of the screen image (light mode, dark mode or night shift mode).
        The color mode is detected by checking the colors of the pixels in the image.

        Args:
            img (ImgBGR): The image to detect the color mode of. Has to be in BGR format.

        Returns:
            ColorMode: The detected color mode.
    """

    DARK_MODE_THRESHOLD = 10 #? Same as in extract_gray
    LIGHT_DARK_THRESHOLD = 75

    img = extract_gray(img, ColorMode.UNKNOWN)

    #? Same as in extract_gray
    gray_mask = (img[:, :, 0] == img[:, :, 1]) & (img[:, :, 1] == img[:, :, 2])

    #? Same as in extract_gray
    dark_mode_mask = (
        ((img[:, :, ColorChannel.BLUE] - img[:, :, ColorChannel.GREEN])
         <= DARK_MODE_THRESHOLD)
        & ((img[:, :, ColorChannel.GREEN] - img[:, :, ColorChannel.RED])
           <= DARK_MODE_THRESHOLD))
    dark_mode_mask &= ~gray_mask

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    light_mode_mask = gray_img > LIGHT_DARK_THRESHOLD
    night_shift_mask = (gray_img < LIGHT_DARK_THRESHOLD) & (gray_img != 0)

    return [ColorMode.LIGHT, ColorMode.DARK, ColorMode.NIGHT_SHIFT][np.argmax(
        [np.mean(mask) for mask in (light_mode_mask, dark_mode_mask, night_shift_mask)])]


def detect_edges(img: ImgGray, iterations=1) -> ImgBool:
    """ Detect the edges in the image using a custom edge detection algorithm.
        Only detect the edges that are lighter on the right (bottom) side than on the
        left (top) side and are the same color as the pixel above (left). Repeat the process
        for iterations times, but drops the requirement of the previous pixel being non-black
        after the first iteration.

        Args:
            img (ImgGray): The image to detect edges in. Has to be a grayscale image.
            iterations (int, optional): The number of iterations to repeat the edge detection.
                Defaults to 1.

        Returns:
            ImgBool: The edges mask.
    """

    vertical, horizontal = img.copy(), img.copy()
    for i in range(iterations):
        lighter_than_left = vertical[1:, 1:] > vertical[1:, :-1]
        same_as_up = vertical[1:, 1:] == vertical[:-1, 1:]
        if i == 0:
            left_non_black = vertical[1:, :-1] != 0
            vertical = lighter_than_left & same_as_up & left_non_black
        else:
            vertical = lighter_than_left & same_as_up

        lighter_than_up = horizontal[1:, 1:] > horizontal[:-1, 1:]
        same_as_left = horizontal[1:, 1:] == horizontal[1:, :-1]
        if i == 0:
            up_non_black = horizontal[:-1, 1:] != 0
            horizontal = lighter_than_up & same_as_left & up_non_black
        else:
            horizontal = lighter_than_up & same_as_left

    edges = vertical | horizontal

    KERNEL1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
    KERNEL2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    BAD_KERNEL1 = np.array([[0, 1, 0], [0, 1, 0], [1, 1, 1]], np.uint8)
    BAD_KERNEL2 = np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1]], np.uint8)
    HIT_OR_MISS_REPS = 2
    for _ in range(HIT_OR_MISS_REPS):
        bad_hit_or_miss1 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, BAD_KERNEL1).astype(bool)
        bad_hit_or_miss2 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, BAD_KERNEL2).astype(bool)
        edges &= ~(bad_hit_or_miss1 | bad_hit_or_miss2)
        hit_or_miss1 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, KERNEL1,
                                       borderType=cv.BORDER_CONSTANT, borderValue=255).astype(bool)
        hit_or_miss2 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, KERNEL2,
                                       borderType=cv.BORDER_CONSTANT, borderValue=255).astype(bool)
        edges = hit_or_miss1 | hit_or_miss2

    larger = np.ones_like(img, bool)
    larger[iterations:, iterations:] = edges.astype(bool)
    return larger


def detect_tiles_grid(img: ImgBGR, color_mode: ColorMode
                      ) -> tp.Tuple[tp.List[int], tp.List[int], int]:
    """ Detect the tiles in the minefield by detecting the vertical and horizontal edges
        of the tiles. See the detect_edges function for more information on the edge detection
        algorithm. The edges are then processed and filtered by size and distance to find the
        grid of the tiles.

        Args:
            img (ImgBGR): The image to detect the tiles in. Has to be in BGR format.
            color_mode (ColorMode): The color mode of the image.

        Returns:
            tp.Tuple[tp.List[int], tp.List[int], int]: The x-coordinates of the vertical edges,
                the y-coordinates of the horizontal edges and the size of the tiles.
    """

    img = cv.cvtColor(extract_gray(img, color_mode), cv.COLOR_BGR2GRAY)
    edges = detect_edges(img, iterations=2).astype(np.uint8) * 255

    MIN_REL_SIZE, MAX_REL_SIZE = 0.9, 1.3
    cnts = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]

    dbg_img = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    # cv.drawContours(dbg_img, cnts, -1, RED, 1)
    cv.imshow('d', dbg_img); cv.waitKey(0); cv.destroyWindow('d')

    edges_x, edges_y, sizes = [], [], []
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 1 and h > 1: continue
        if w > 1: sizes.append(w)
        if h > 1: sizes.append(h)

    median_size = np.median(sizes)
    max_size = median_size * MAX_REL_SIZE

    min_size, max_size = median_size * MIN_REL_SIZE, median_size * MAX_REL_SIZE

    dbg_img = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 1 and h > 1: cv.drawContours(dbg_img, [cnt], -1, (255,0,0), 1); continue
        if min_size <= w <= max_size: edges_y.append(y); cv.drawContours(dbg_img, [cnt], -1, RED, 1)
        if min_size <= h <= max_size: edges_x.append(x); cv.drawContours(dbg_img, [cnt], -1, (0,255,0), 1)

    cv.imshow('d', dbg_img); cv.waitKey(0); cv.destroyWindow('d')

    sorted_x = np.unique(edges_x)
    sorted_y = np.unique(edges_y)
    if len(sorted_x) < 2 or len(sorted_y) < 2:
        return [], [], -1 # Not enough edges found

    SAME_THRESHOLD = 0.35
    sorted_x = sorted_x[np.diff(sorted_x, prepend=-median_size) > (SAME_THRESHOLD * median_size)]
    sorted_y = sorted_y[np.diff(sorted_y, prepend=-median_size) > (SAME_THRESHOLD * median_size)]

    CLOSE_THRESHOLD, FAR_THRESHOLD = 0.7, 1.3
    median_diff = np.median(np.concatenate([np.diff(sorted_x), np.diff(sorted_y)]))
    min_size, max_size = CLOSE_THRESHOLD * median_diff, FAR_THRESHOLD * median_diff
    while len(sorted_x) > 1 and not (min_size <= (sorted_x[-1] - sorted_x[-2]) <= max_size):
        sorted_x = sorted_x[:-1]
    while len(sorted_y) > 1 and not (min_size <= (sorted_y[-1] - sorted_y[-2]) <= max_size):
        sorted_y = sorted_y[:-1]
    while len(sorted_x) > 1 and not (min_size <= (sorted_x[1] - sorted_x[0]) <= max_size):
        sorted_x = sorted_x[1:]
    while len(sorted_y) > 1 and not (min_size <= (sorted_y[1] - sorted_y[0]) <= max_size):
        sorted_y = sorted_y[1:]

    if len(sorted_x) < 2 or len(sorted_y) < 2:
        return [], [], -1 # Not enough suitable edges found

    median_diff = int(np.median(np.concatenate([np.diff(sorted_x), np.diff(sorted_y)])))

    if sorted_x[-1] + median_diff > np.shape(img)[1]:
        sorted_x = sorted_x[:-1]
    if sorted_y[-1] + median_diff > np.shape(img)[0]:
        sorted_y = sorted_y[:-1]

    GOOD_ENOUGH_THRESHOLD = 0.1
    good_enough_diff = median_diff * GOOD_ENOUGH_THRESHOLD
    left, right = sorted_x[0], sorted_x[-1]
    top, bottom = sorted_y[0], sorted_y[-1]
    n_cols, n_rows = len(sorted_x) - 1, len(sorted_y) - 1
    orig_sorted_x, orig_sorted_y = sorted_x.copy(), sorted_y.copy()
    if abs((right - left) - n_cols * median_diff) <= good_enough_diff:
        sorted_x = np.arange(left, left + n_cols * median_diff + 1, median_diff)
        while (np.mean(sorted_x - orig_sorted_x) > 0.5
               and orig_sorted_x[0] - sorted_x[0] <= good_enough_diff):
            sorted_x -= 1
        while (np.mean(orig_sorted_x - sorted_x) > 0.5
               and sorted_x[-1] - orig_sorted_x[-1] <= good_enough_diff):
            sorted_x += 1
    if abs((bottom - top) - n_rows * median_diff) <= good_enough_diff:
        sorted_y = np.arange(top, top + n_rows * median_diff + 1, median_diff)
        while (np.mean(sorted_y - orig_sorted_y) > 0.5
               and orig_sorted_y[0] - sorted_y[0] <= good_enough_diff):
            sorted_y -= 1
        while (np.mean(orig_sorted_y - sorted_y) > 0.5
               and sorted_y[-1] - orig_sorted_y[-1] <= good_enough_diff):
            sorted_y += 1

    return list(sorted_x), list(sorted_y), median_diff


def match_colors(img: ImgBGR, color_thresh_dict: Thresholds,
                 expected_state_values: tp.Dict[int, int]) -> ImgGray:
    """ Match the colors of pixels in the image to the colors in the color threshold dictionary.

        Args:
            img (ImgBGR): The image to match the colors in. Has to be in BGR format.
            color_thresh_dict (Thresholds): The color threshold dictionary in the form
                {state: [(lower_bound, upper_bound), ...]}. The state is the value that is
                assigned to the pixel if the color matches the threshold. If a pixel doesn't match
                any of the thresholds, it is assigned the value UNMATCHED. If a pixel matches
                multiple thresholds, the state with the closest expected value to the pixel's
                expected value assigned to the pixel.


        Returns:
            ImgGray: The image with the colors matched to the states.
    """

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    matched_img = np.full(np.shape(hsv_img)[:2], UNMATCHED, np.uint8)
    total_mask = np.zeros(np.shape(hsv_img)[:2], bool)
    for state, bounds in color_thresh_dict.items():
        # mask = cv.inRange(hsv_img, lower_bound, upper_bound)
        mask = np.bitwise_or.reduce([cv.inRange(hsv_img, lower_bound, upper_bound)
                                     for lower_bound, upper_bound in bounds])
        mask = mask.astype(bool)
        matched_img[mask & ~total_mask] = state
        colliding_mask = np.bitwise_and(mask, total_mask)
        real_values = hsv_img[colliding_mask][:, 2]
        expected_values_for_prev_matched = np.zeros_like(real_values)
        for prev_state in color_thresh_dict:
            if prev_state == state: continue
            expected_values_for_prev_matched[matched_img[colliding_mask] == prev_state] \
                = expected_state_values[prev_state]
        matched_img[colliding_mask] = np.where(
            (np.abs(real_values - expected_state_values[state])
             < np.abs(real_values - expected_values_for_prev_matched)),
            state,
            matched_img[mask & total_mask])
        total_mask |= mask

    return matched_img


def get_brightness(img: ImgBGR, color_mode: ColorMode, guess: int) -> int:
    """ Get the brightness of the screen image. The brightness is calculated by finding the
        unclicked and empty tiles in the minefield and calculating the average brightness of
        the pixels in the tiles, then comparing the brightness of the tiles to the maximum
        brightness of the tiles in the color mode and averaging the two.

        Args:
            img (ImgBGR): The image to get the brightness of. Has to be in BGR format.
            color_mode (ColorMode): The color mode of the image.
            guess (int): The guess of the brightness of the screen. In [50, 100].

        Returns:
            int: The brightness of the screen image in [0, 100].
    """

    VALUE_THRESHOLD = 2
    thresholds_high = state_thresholds(color_mode, 100, value_threshold=VALUE_THRESHOLD)
    thresholds_low = state_thresholds(color_mode, 50, value_threshold=VALUE_THRESHOLD)
    real_mid_colors = state_thresholds(color_mode, guess, no_thresholds=True)
    real_max_colors = state_thresholds(color_mode, 100, no_thresholds=True)
    expected_state_values = {state: real_mid_colors[state][0][0][2] for state in real_mid_colors}

    thresholds: Thresholds = {}
    for state in (UNCLICKED, EMPTY):
        sat_low, sat_high = thresholds_low[state][0][0][1], thresholds_low[state][0][1][1]
        val_low, val_high = thresholds_low[state][0][0][2], thresholds_high[state][0][1][2]
        if len(thresholds_low[state]) == 1:
            hue_low, hue_high = thresholds_low[state][0][0][0], thresholds_low[state][0][1][0]
            thresholds[state] = (((hue_low, sat_low, val_low), (hue_high, sat_high, val_high)),)
        else:
            hue_low1, hue_high1 = thresholds_low[state][0][0][0], thresholds_low[state][0][1][0]
            hue_low2, hue_high2 = thresholds_low[state][1][0][0], thresholds_low[state][1][1][0]
            thresholds[state] = (((hue_low1, sat_low, val_low), (hue_high1, sat_high, val_high)),
                                 ((hue_low2, sat_low, val_low), (hue_high2, sat_high, val_high)))
    matched_img = match_colors(img, thresholds, expected_state_values)
    unclicked_mask = matched_img == UNCLICKED
    empty_mask = matched_img == EMPTY
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    MASK_THRESHOLD = .1
    if np.mean(unclicked_mask) > MASK_THRESHOLD:
        unclicked_median_value = np.median(hsv_img[unclicked_mask][:, 2])
    else:
        unclicked_median_value = -1
    unclicked_brightness = unclicked_median_value / real_max_colors[UNCLICKED][0][1][2]
    if np.mean(empty_mask) > MASK_THRESHOLD:
        empty_median_value = np.median(hsv_img[empty_mask][:, 2])
    else:
        empty_median_value = -1
    empty_brightness = empty_median_value / real_max_colors[EMPTY][0][1][2]

    if unclicked_brightness < 0 and empty_brightness < 0:
        print('ERROR: Failed to detect brightness')

    brightness = int(max(0, min(1,
                      np.mean([brightness for brightness in
                               (unclicked_brightness, empty_brightness)
                               if brightness > 0])
    )) * 100)
    return brightness


def detect_tiles_states(img: ImgBGR, rect_grid: RectGrid, color_mode: ColorMode,
                        brightness: int) -> GridState:
    """ Detect the states of the tiles in the minefield. The states are detected by checking
        the color of the tile. TODO

        Args:
            img (ImgBGR): The image of the minefield. Has to be in BGR format.
            rect_grid (RectGrid): The grid of the rectangles of the tiles.
            color_mode (ColorMode): The color mode of the image.
            brightness (int): The brightness of the screen. In [50, 100].

        Returns:
            GridState: The states of the tiles.
    """

    n_rows, n_cols = len(rect_grid), len(rect_grid[0])
    grid: Uint8_2D = np.ndarray((n_rows, n_cols), np.uint8)

    expected_colors = state_thresholds(color_mode, brightness, no_thresholds=True)
    expected_state_values = {state: expected_colors[state][0][0][2] for state in expected_colors}
    matched: ImgGray = match_colors(img, state_thresholds(color_mode, brightness),
                                    expected_state_values)
    # only flagged tiles have a majority of unmatched pixels
    matched[matched == UNMATCHED] = FLAGGED

    STATE_THRESHOLD = 0.2
    FLAGGED_THRESHOLD = 0.15
    MIN_MIN_HALF_OVER_MAX = 0.1
    REL_PADDING = 0.2
    for r in range(n_rows):
        for c in range(n_cols):
            # Crop to the tile and keep both the full tile and the inner part of the tile
            (left, top), (right, bottom) = rect_grid[r][c]
            width, height = right - left, bottom - top
            full_tile = matched[top:bottom, left:right]
            full_size = width * height
            top += int(REL_PADDING * height)
            left += int(REL_PADDING * width)
            bottom -= int(REL_PADDING * height)
            right -= int(REL_PADDING * width)
            width, height = right - left, bottom - top
            tile = matched[top:bottom, left:right]
            size = width * height

            # Get counts of each state in the inner part of the tile
            counts = np.bincount(tile.flatten(), minlength=256)

            # If enough pixels are matched to a number, set the state to that number
            state = int(np.argmax(counts[1:9]) + 1
                     if np.max(counts[1:9]) > STATE_THRESHOLD * size
                     else np.argmax(counts[:UNMATCHED]))

            # If enough pixels are possibly flagged and no number was found, consider it flagged
            if counts[FLAGGED] > FLAGGED_THRESHOLD * size and not (0 < state < 9):
                state = FLAGGED

            # If the halves of the tile have different states, consider it flagged
            up_half = np.mean(tile[:height//2] == state)
            down_half = np.mean(tile[height//2:] == state)
            if min(up_half, down_half) < MIN_MIN_HALF_OVER_MAX * max(up_half, down_half):
                state = FLAGGED

            EMPTY_BORDER_THRESHOLD = 0.5
            if color_mode == ColorMode.LIGHT:
                # Determine if the tile is empty or unclicked based on its border
                if state in (0, 9):
                    full_zeros = np.count_nonzero(full_tile == 0)
                    inner_zeros = counts[0]
                    border_zeros = full_zeros - inner_zeros
                    border_size = full_size - size
                    if border_zeros > EMPTY_BORDER_THRESHOLD * border_size:
                        state = 0
                    else:
                        state = 9

            grid[r, c] = state

    return grid


###################################################################
#                        MAIN CLASS AND GUI                       #
###################################################################

class Minesweeper:
    def __init__(self):
        self._setup()
        self._main_loop()

    def get_screen(self, area: tp.Optional[Rect] = None, mask_window: bool = False) -> ImgBGR:
        """ Get the screen image, optionally masking the window of this program with black 
            and/or cropping to a specific area.

            Args:
                area (tp.Optional[Rect], optional): The area to crop to in the form of
                    ((left, top), (right, bottom)). Defaults to None.
                mask_window (bool, optional): Whether to mask the window of this program with black.
                    Defaults to False.

            Returns:
                ImgBGR: The screen image.
        """

        screen = cv.cvtColor(np.array(pyautogui.screenshot()), cv.COLOR_RGB2BGR)

        # Cut out (fill with black) the window of this program
        win_pos = cv.getWindowImageRect(WINDOW_NAME)
        left, top, width, height = win_pos
        right, bottom = left + width, top + height
        if width > 0 and height > 0:
            screen[top:bottom, left:right] = 0

        if area is not None:
            (left, top), (right, bottom) = area
            screen = screen[top:bottom, left:right]

        return screen

    def reset_selection(self):
        """ Resets the selected screen area to show the whole screen
            and select the minefield again.
        """

        self.selection_corner: Pos = (0, 0)
        self.selected_rect: Rect = NONE_RECT
        self.minefield_rect_in_selection: Rect = NONE_RECT
        self.minefield_rect_global: Rect = NONE_RECT
        self.color_mode: ColorMode = ColorMode.UNKNOWN
        self.is_grid_detected: bool = False

    def _setup(self):
        """ Initialize the GUI and the state of the program. """

        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        self.reset_selection()
        cv.setMouseCallback(WINDOW_NAME, self._handle_mouse_event) # type: ignore

    def _handle_mouse_event(self, event, *_):
        """ Handle mouse events in the GUI. Used to select the minefield area. """

        if event in (cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONUP):
            mouse_x, mouse_y = pyautogui.position()
            screen_width, screen_height = pyautogui.size()
            win_left, win_top, win_width, win_height = cv.getWindowImageRect(WINDOW_NAME)
            rel_x, rel_y = (mouse_x - win_left) / win_width, (mouse_y - win_top) / win_height
            screen_x, screen_y = int(screen_width * rel_x), int(screen_height * rel_y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.selection_corner = (screen_x, screen_y)
        elif event == cv.EVENT_LBUTTONUP:
            if (abs(screen_x - self.selection_corner[0]) > 10 and
                    abs(screen_y - self.selection_corner[1]) > 10):
                self.selected_rect = (self.selection_corner, (screen_x, screen_y))
                self.handle_selected_rect_change()

    def handle_selected_rect_change(self):
        """ Handle the change of the selected screen area.
            Detects the color mode and finds the minefield and the grid of tiles.
        """

        (left, top), (right, bottom) = self.selected_rect
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        self.selected_rect = ((left, top), (right, bottom))

        screen: ImgBGR = self.get_screen(self.selected_rect)

        # Find the minefield in the selection
        self.minefield_rect_in_selection = find_minefield(screen)
        (s_left, s_top), (s_right, s_bottom) = self.selected_rect
        (m_left, m_top), (m_right, m_bottom) = self.minefield_rect_in_selection
        self.minefield_rect_global = ((s_left + m_left, s_top + m_top),
                                      (s_left + m_right, s_top + m_bottom))

        screen = self.get_screen(self.minefield_rect_global)

        # Detect the color mode
        self.color_mode = detect_color_mode(screen)

        # Find the grid of tiles
        self.vertical_lines, self.horizontal_lines, self.tile_size \
            = detect_tiles_grid(screen, self.color_mode)
        self.n_cols = len(self.vertical_lines)
        self.n_rows = len(self.horizontal_lines)
        self.is_grid_detected = self.tile_size != -1

        # Create the grid of rectangles of the tiles
        x = self.vertical_lines + [self.vertical_lines[-1] + self.tile_size]
        y = self.horizontal_lines + [self.horizontal_lines[-1] + self.tile_size]
        n_cols, n_rows = len(self.vertical_lines), len(self.horizontal_lines)
        self.rect_grid = [
            [
                ((x[c], y[r]), (x[c+1], y[r+1])) for c in range(n_cols)
            ] for r in range(n_rows)
        ]

        # Detect the brightness of the screen
        BRIGHTNESS_RETRIES = 5
        START_BRIGHTNESS = 75
        self.brightness = START_BRIGHTNESS
        for _ in range(BRIGHTNESS_RETRIES):
            self.brightness = self.get_grid_brightness(screen)

    def get_grid_brightness(self, screen: ImgBGR) -> int:
        """ Get the brightness of the grid of tiles in the screen image.

        Args:
            screen (ImgBGR): The screen image.

        Returns:
            int: The brightness of the grid of tiles.
        """

        grid_screen = screen.copy()

        mask = np.zeros_like(grid_screen, bool)
        REL_PADDING = 0.2
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                (left, top), (right, bottom) = self.rect_grid[r][c]
                width, height = right - left, bottom - top
                top += int(REL_PADDING * height)
                left += int(REL_PADDING * width)
                bottom -= int(REL_PADDING * height)
                right -= int(REL_PADDING * width)
                mask[top:bottom, left:right] = True
        grid_screen[np.logical_not(mask)] = 0

        left, top = self.rect_grid[0][0][0]
        right, bottom = self.rect_grid[-1][-1][1]
        grid_screen = screen[top:bottom, left:right]

        return get_brightness(grid_screen, self.color_mode, self.brightness)


    def draw_debug_grid(self, screen: ImgBGR):
        """ Draw the detected grid of tiles on the screen image for debugging purposes.

        Args:
            screen (ImgBGR): The screen image to draw on.
        """

        if self.is_grid_detected:
            left, top = self.vertical_lines[0], self.horizontal_lines[0]
            right = self.vertical_lines[-1] + self.tile_size
            bottom = self.horizontal_lines[-1] + self.tile_size
            for x in self.vertical_lines:
                cv.line(screen, (x, top), (x, bottom), RED, 2)
            cv.line(screen, (right, top), (right, bottom), RED, 2)
            for y in self.horizontal_lines:
                cv.line(screen, (left, y), (right, y), RED, 2)
            cv.line(screen, (left, bottom), (right, bottom), RED, 2)
            for r in range(self.n_rows):
                for c in range(self.n_cols):
                    pos = (self.rect_grid[r][c][0][0] + int(self.tile_size / 5),
                           self.rect_grid[r][c][0][1] + int(self.tile_size * 0.8))
                    cv.putText(screen, str(self.grid[r, c]), pos, cv.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2)
        else:
            print('NO GRID DETECTED')

    def _main_loop(self):
        """ The main loop of the program. """

        target_delta_time = 1e9 / MAX_FPS
        prev_time = time_ns()
        while True:
            if not self.is_grid_detected:
                screen = self.get_screen(mask_window=True)
            else:
                screen = self.get_screen(self.minefield_rect_global)
                self.brightness = self.get_grid_brightness(screen)
                self.grid = detect_tiles_states(screen, self.rect_grid,
                                                self.color_mode, self.brightness)
                self.draw_debug_grid(screen) # debug

            # Show the screen
            cv.imshow(WINDOW_NAME, screen)

            # Limit the frame rate and handle user input
            curr_time = time_ns()
            delta_time = curr_time - prev_time
            prev_time = curr_time
            wait_time = max(1, int(target_delta_time - delta_time) // 10**6)
            key = cv.waitKey(wait_time)
            if key == ord('q'):
                break # Quit
            elif key == ord('r'):
                self.reset_selection()

        cv.destroyAllWindows()


if __name__ == '__main__':
    print(
        'Welcome to Minesweeper Cheat!\n'
        '1. Click and drag to select the minefield area.\n'
        '2. Press "r" to reset the selection.\n'
        '3. Press "q" to quit.\n'
        'If the game state is not detected properly, try adjusting the brightness \
         (potentially adjusting it to the original state in small steps) \
         or the zoom level of the game, or try reselecting the minefield area.\n'
        'Low resolution mode may work, but also may not, especially with low zoom levels.\n'
    )
    Minesweeper()