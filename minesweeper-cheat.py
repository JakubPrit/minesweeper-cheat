import cv2 as cv
import numpy as np
import pyautogui
import typing as tp
from time import time_ns
from scipy.stats import mode # type: ignore
from enum import Enum
from math import isclose


###################################################################
#                  TYPE DEFINITIONS AND CONSTANTS                 #
###################################################################

Pos = tp.Tuple[int, int]
Rect = tp.Tuple[Pos, Pos]
Img = np.ndarray
HSVColor = tp.Tuple[int, int, int]


WINDOW_NAME = 'Minesweeper Cheat'
MAX_FPS = 5
NONE_RECT = ((-1, -1), (-1, -1))
BLACK = (0, 0, 0)
RED = (0, 0, 255)


EMPTY = 0
UNCLICKED = 9
FLAGGED = 10
UNMATCHED = 255


# HSV input format: hue in [0, 360], saturation in [0, 100], value in [0, 100]
LIGHT_MODE_STATE_HSV_COLORS = {
    0: (0, 0, 78),
    1: (240, 100, 100),
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
    UNCLICKED: (210, 18, 35),
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


def colors_to_thresholds(colors: tp.Dict[int, HSVColor]
                         ) -> tp.Dict[int, tp.Tuple[HSVColor, HSVColor]]:
    # Output HSV format: hue in [0, 180], saturation in [0, 255], value in [0, 255]

    return {
        key: (
            (
                max(0, (color[0] // 2) - HUE_THRESHOLD),
                max(0, (color[1] * 255 // 100) - SATURATION_THRESHOLD),
                max(0, (color[2] * 255 // 100) // 2 - VALUE_THRESHOLD)
            ), (
                min(101, (color[0] // 2) // 2 + HUE_THRESHOLD),
                min(101, (color[1] * 255 // 100) + SATURATION_THRESHOLD),
                min(101, (color[2] * 255 // 100) + VALUE_THRESHOLD)
            )
        )
        for key, color in colors.items()
    }


HUE_THRESHOLD = 10
SATURATION_THRESHOLD = 10
VALUE_THRESHOLD = 5
LIGHT_MODE_STATE_HSV_THRESHOLDS = colors_to_thresholds(LIGHT_MODE_STATE_HSV_COLORS)
DARK_MODE_STATE_HSV_THRESHOLDS = colors_to_thresholds(DARK_MODE_STATE_HSV_COLORS)
NIGHT_SHIFT_STATE_HSV_THRESHOLDS = colors_to_thresholds(NIGHT_SHIFT_STATE_HSV_COLORS)


class ColorChannel:
    RED = 2
    GREEN = 1
    BLUE = 0


class ColorMode(Enum):
    LIGHT = 0
    DARK = 1
    NIGHT_SHIFT = 2


###################################################################
#                  MINESWEEPER STATE RECOGNITION                  #
###################################################################

def find_minefield(screen: Img) -> Rect:
    """ Find the minefield in the screen image by detecting the largest rectangle.

    Args:
        screen (Img): The screen image.

    Returns:
        Rect: The rectangle of the minefield in the form ((left, top), (right, bottom)).
    """

    edges = cv.Canny(screen, 100, 200)
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


def extract_gray(img: Img) -> Img:
    """ Mask out all non-gray pixels in the image. Gray pixels are defined as pixels
        that have the same value in all three color channels, or pixels that have their blue
        channel value slightly higher than the green channel value and the red channel value
        slightly lower than the green channel value, with these differences not higher than
        DARK_MODE_THRESHOLD (corresponds to minesweeper.online dark mode tile colors).

    Args:
        bgr_img (Img): The image to mask. Has to be in BGR format.

    Returns:
        Img: The masked image.
    """

    BLACK_THRESHOLD = 20
    WHITE_THRESHOLD = 240
    DARK_MODE_THRESHOLD = 8 #? Same as in detect_color_mode

    # Handle exactly gray pixels (not blueish)
    #? Same as in detect_color_mode
    mask = (img[:, :, 0] == img[:, :, 1]) & (img[:, :, 1] == img[:, :, 2])

    # Handle blueish (dark mode) pixels
    #? Same as in detect_color_mode
    mask |= (((img[:, :, ColorChannel.BLUE] - img[:, :, ColorChannel.GREEN])
              <= DARK_MODE_THRESHOLD)
             & ((img[:, :, ColorChannel.GREEN] - img[:, :, ColorChannel.RED])
                <= DARK_MODE_THRESHOLD))

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask &= (img > BLACK_THRESHOLD) & (img < WHITE_THRESHOLD)

    img = cv.bitwise_and(img, img, mask=mask.astype(np.uint8))
    return img


def detect_color_mode(img: Img) -> ColorMode:
    DARK_MODE_THRESHOLD = 8 #? Same as in extract_gray
    LIGHT_DARK_THRESHOLD = 120

    img = extract_gray(img)

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
    night_shift_mask = (gray_img < LIGHT_DARK_THRESHOLD) & (gray_img != BLACK)

    return [ColorMode.LIGHT, ColorMode.DARK, ColorMode.NIGHT_SHIFT][np.argmax(
        [np.mean(mask) for mask in (light_mode_mask, dark_mode_mask, night_shift_mask)])]


def detect_edges(img: Img, vertical: bool, iterations=1) -> Img:
    """ Detect the edges in the image using a custom edge detection algorithm.
        Only detect the edges that are lighter on the right (bottom) side than on the
        left (top) side and are the same color as the pixel above (left). Repeat the process
        for iterations times, but drops the requirement of the previous pixel being non-black
        after the first iteration.

    Args:
        img (Img): The image to detect edges in.
        vertical (bool): Whether to detect vertical edges (True) or horizontal edges (False).
        iterations (int, optional): The number of iterations to repeat the edge detection.
            Defaults to 1.

    Returns:
        Img: The edges mask.
    """

    for i in range(iterations):
        if vertical:
            lighter_than_left = img[1:, 1:] > img[1:, :-1]
            same_as_up = img[1:, 1:] == img[:-1, 1:]
            if i == 0:
                left_non_black = img[1:, :-1] != 0
                img = lighter_than_left & same_as_up & left_non_black
            else:
                img = lighter_than_left & same_as_up
        else:
            lighter_than_up = img[1:, 1:] > img[:-1, 1:]
            same_as_left = img[1:, 1:] == img[1:, :-1]
            if i == 0:
                up_non_black = img[:-1, 1:] != 0
                img = lighter_than_up & same_as_left & up_non_black
            else:
                img = lighter_than_up & same_as_left
    return img


def detect_tiles_grid(img: Img) -> tp.Tuple[tp.List[int], tp.List[int], int]:
    """ Detect the tiles in the minefield by detecting the vertical and horizontal edges
        of the tiles. See the detect_edges function for more information on the edge detection
        algorithm. The edges are then processed and filtered by size and distance to find the
        grid of the tiles.

    Args:
        img (Img): The image to detect the tiles in.

    Returns:
        tp.Tuple[tp.List[int], tp.List[int], int]: The x-coordinates of the vertical edges,
            the y-coordinates of the horizontal edges and the size of the tiles.
    """

    img = extract_gray(img)
    vertical_mask = detect_edges(img, vertical=True, iterations=2)
    horizontal_mask = detect_edges(img, vertical=False, iterations=2)
    edges = horizontal_mask | vertical_mask

    KERNEL1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], np.uint8)
    KERNEL2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], np.uint8)
    HIT_OR_MISS_REPS = 1
    for _ in range(HIT_OR_MISS_REPS):
        hit_or_miss1 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, KERNEL1)
        hit_or_miss2 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, KERNEL2)
        edges = hit_or_miss1 | hit_or_miss2

    MIN_REL_SIZE, MAX_REL_SIZE = 0.75, 1.3
    cnts = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    edges_x, edges_y, sizes = [], [], []
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 1 and h > 1: continue
        if w > 1: sizes.append(w)
        if h > 1: sizes.append(h)
    median_size = np.median(sizes)
    min_size, max_size = median_size * MIN_REL_SIZE, median_size * MAX_REL_SIZE
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 1 and h > 1: continue
        if min_size <= w <= max_size: edges_y.append(y)
        if min_size <= h <= max_size: edges_x.append(x)

    sorted_x = np.unique(edges_x)
    sorted_y = np.unique(edges_y)
    if len(sorted_x) < 2 or len(sorted_y) < 2:
        return [], [], -1 # Not enough edges found

    SAME_THRESHOLD = 0.3
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
    while len(sorted_x) > 1 and sorted_x[0] < (CLOSE_THRESHOLD * median_diff):
        sorted_x = sorted_x[1:]
    while len(sorted_y) > 1 and sorted_y[0] < (CLOSE_THRESHOLD * median_diff):
        sorted_y = sorted_y[1:]

    if len(sorted_x) < 2 or len(sorted_y) < 2:
        return [], [], -1 # Not enough suitable edges found

    # // # Debug
    # // if cv.waitKey(1) == ord('d'):
    # //     dbg_img = edges.astype(np.uint8) * 255
    # //     contours = cv.findContours(dbg_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    # //     dbg_img = cv.cvtColor(dbg_img, cv.COLOR_GRAY2BGR)
    # //     for cnt in contours:
    # //         x, y, w, h = cv.boundingRect(cnt)
    # //         cv.rectangle(dbg_img, (x, y), (x + w, y + h), (0, 255, 0), 1)
    # //     cv.imshow('debug', dbg_img)
    # //     cv.waitKey(0)
    # //     cv.destroyWindow('debug')

    median_diff = np.median(np.concatenate([np.diff(sorted_x), np.diff(sorted_y)]))
    return sorted_x, sorted_y, int(median_diff)


def match_colors(img: Img, color_thresh_dict: tp.Dict[int, tp.Tuple[HSVColor, HSVColor]]) -> Img:
    """ Match the colors of pixels in the image to the colors in the color threshold dictionary.

    Args:
        img (Img): The image to match the colors in. Should be in standard BGR format.
        color_thresh_dict (tp.Dict[int, tp.Tuple[HSVColor, HSVColor]]): The color threshold
            dictionary in the form {state: (lower_bound, upper_bound)}. The state is the value
            that the pixel is assigned to if the color matches the threshold.

    Returns:
        Img: The image with the colors matched to the states.
    """

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('debug orig', cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)); cv.waitKey(0); cv.destroyWindow('debug orig')
    matched_img = np.full_like(img, UNMATCHED)
    total_mask = np.zeros(np.shape(hsv_img)[:2], bool)
    for state, (lower_bound, upper_bound) in color_thresh_dict.items():
        mask = cv.inRange(hsv_img, lower_bound, upper_bound)
        cv.imshow('debug{}'.format(state), mask); cv.waitKey(0); cv.destroyWindow('debug{}'.format(state))
        mask = mask.astype(bool)
        matched_img[mask] = state
        total_mask |= mask
    cv.imshow('debug all', total_mask.astype(np.uint8) * 255); cv.waitKey(0); cv.destroyWindow('debug all')

    return matched_img


def detect_tiles_states(img: Img, vertical_lines: tp.List[int], horizontal_lines: tp.List[int],
                        tile_size: int, color_mode: ColorMode) -> tp.List[tp.List[bool]]:
    """ Detect the states of the tiles in the minefield. The states are detected by checking
        the color of the tile. TODO

    Args:
        img (Img): The image of the minefield.
        vertical_lines (tp.List[int]): The x-coordinates of the vertical edges of the tiles.
        horizontal_lines (tp.List[int]): The y-coordinates of the horizontal edges of the tiles.
        tile_size (int): The size of the tiles.

    Returns:
        tp.List[tp.List[TileState]]: The states of the tiles.
    """

    all_thresholds = {
        ColorMode.LIGHT : LIGHT_MODE_STATE_HSV_THRESHOLDS,
        ColorMode.DARK : DARK_MODE_STATE_HSV_THRESHOLDS,
        ColorMode.NIGHT_SHIFT : NIGHT_SHIFT_STATE_HSV_THRESHOLDS
    }

    # DEBUG
    dbg_img = match_colors(img, all_thresholds[color_mode])
    cv.imshow('dbg', dbg_img)
    cv.waitKey(0)
    cv.destroyWindow('dbg')
    return [[]]

    raise NotImplementedError


###################################################################
#                        MAIN CLASS AND GUI                       #
###################################################################

class Minesweeper:
    def __init__(self):
        self._setup()
        self._main_loop()

    def get_screen(self, area: tp.Optional[Rect] = None, mask_window: bool = False) -> Img:
        """ Get the screen image, optionally masking the window of this program with black 
            and/or cropping to a specific area.


        Args:
            area (tp.Optional[Rect], optional): The area to crop to in the form of
                ((left, top), (right, bottom)). Defaults to None.
            mask_window (bool, optional): Whether to mask the window of this program with black.
                Defaults to False.

        Returns:
            Img: The screen image.
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

    def _setup(self):
        """ Initialize the GUI and the state of the program. """

        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        self.is_grid_detected = False
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
            Finds the minefield in the selection.
        """

        (left, top), (right, bottom) = self.selected_rect
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        self.selected_rect = ((left, top), (right, bottom))

        screen: Img = self.get_screen(self.selected_rect)

        # Find the color mode
        self.color_mode = detect_color_mode(screen)

        # Find the minefield in the selection
        self.minefield_rect_in_selection = find_minefield(screen)
        (s_left, s_top), (s_right, s_bottom) = self.selected_rect
        (m_left, m_top), (m_right, m_bottom) = self.minefield_rect_in_selection
        self.minefield_rect_global = ((s_left + m_left, s_top + m_top),
                                      (s_left + m_right, s_top + m_bottom))

        screen = self.get_screen(self.minefield_rect_global)

        # Find the grid of tiles
        self.vertical_lines, self.horizontal_lines, self.tile_size = detect_tiles_grid(screen)
        self.n_cols = len(self.vertical_lines)
        self.n_rows = len(self.horizontal_lines)
        self.is_grid_detected = self.tile_size != -1

    # Debug
    def draw_debug_grid(self, screen: Img):
        """ Draw the detected grid of tiles on the screen image for debugging purposes.

        Args:
            screen (Img): The screen image to draw on.
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
        else:
            print('NO GRID DETECTED')

    def _main_loop(self):
        """ The main loop of the program. """

        target_delta_time = 1e9 / MAX_FPS
        prev_time = time_ns()
        while True:
            if self.selected_rect == NONE_RECT:
                screen = self.get_screen(mask_window=True)
            else:
                screen = self.get_screen(self.minefield_rect_global)
                self.draw_debug_grid(screen)
                detect_tiles_states(screen, self.vertical_lines, self.horizontal_lines,
                                    self.tile_size, self.color_mode)

            # // # Detect the grid and tiles
            # // # TODO: Don't do this every frame, just in handle_selected_rect_change, THIS IS JUST FOR TESTING
            # // vertical_lines, horizontal_lines, tile_size = detect_tiles(screen)

            # // # debug
            # // # print(vertical_lines, horizontal_lines, tile_size)
            # // screen = extract_gray(screen)
            # // # Set all non-masked pixels to last masked color left of them
            # // for x in vertical_lines:
            # //     cv.line(screen, (x, 0), (x, screen.shape[0]), (0, 0, 255), 2)
            # // for y in horizontal_lines:
            # //     cv.line(screen, (0, y), (screen.shape[1], y), (0, 0, 255), 2)
            # // cv.rectangle(screen, (0, 0), (tile_size, tile_size), (255, 0, 0), 2)


            # Show the screen
            cv.imshow(WINDOW_NAME, screen)

            # TODO: Detect game state

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
    Minesweeper()