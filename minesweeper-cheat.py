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


WINDOW_NAME = 'Minesweeper Cheat'
MAX_FPS = 5
NONE_RECT = ((-1, -1), (-1, -1))
BLACK = (0, 0, 0)
RED = (0, 0, 255)


class ColorChannel:
    RED = 2
    GREEN = 1
    BLUE = 0


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


def extract_gray(bgr_img: Img) -> Img:
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

    BLACK_WHITE_THRESHOLD = 100
    DARK_MODE_THRESHOLD = 8

    # Handle exactly gray pixels (not blueish)
    mask = (bgr_img[:, :, 0] == bgr_img[:, :, 1]) & (bgr_img[:, :, 1] == bgr_img[:, :, 2])

    # Handle blueish (dark mode) pixels
    mask |= (((bgr_img[:, :, ColorChannel.BLUE] - bgr_img[:, :, ColorChannel.GREEN])
              <= DARK_MODE_THRESHOLD)
             & ((bgr_img[:, :, ColorChannel.GREEN] - bgr_img[:, :, ColorChannel.RED])
                <= DARK_MODE_THRESHOLD))

    bgr_img = cv.cvtColor(bgr_img, cv.COLOR_BGR2GRAY)
    avg_color = np.mean(bgr_img[mask])
    mask &= (np.abs(bgr_img - avg_color) < BLACK_WHITE_THRESHOLD)

    bgr_img = cv.bitwise_and(bgr_img, bgr_img, mask=mask.astype(np.uint8))
    return bgr_img


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


def detect_tiles(img: Img) -> tp.Tuple[tp.List[int], tp.List[int], int]:
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

    KERNEL1 = np.array([[0, 0, 0],
                        [1, 1, 1],
                        [0, 0, 0]], np.uint8)
    KERNEL2 = np.array([[0, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]], np.uint8)
    HIT_OR_MISS_REPS = 1
    for _ in range(HIT_OR_MISS_REPS):
        hit_or_miss1 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, KERNEL1)
        hit_or_miss2 = cv.morphologyEx(edges.astype(np.uint8), cv.MORPH_HITMISS, KERNEL2)
        edges = hit_or_miss1 | hit_or_miss2

    MIN_REL_SIZE = 0.75
    MAX_REL_SIZE = 1.3
    cnts = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)[0]
    edges_x, edges_y = [], []
    sizes = []
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 1 and h > 1:
            continue
        if w > 1: sizes.append(w)
        if h > 1: sizes.append(h)
    median_size = np.median(sizes)
    min_size = median_size * MIN_REL_SIZE
    max_size = median_size * MAX_REL_SIZE
    for cnt in cnts:
        x, y, w, h = cv.boundingRect(cnt)
        if w > 1 and h > 1:
            continue
        if min_size <= w <= max_size:
            edges_y.append(y)
        if min_size <= h <= max_size:
            edges_x.append(x)

    sorted_x = np.unique(edges_x)
    sorted_y = np.unique(edges_y)
    if len(sorted_x) < 2 or len(sorted_y) < 2:
        return [], [], -1 # Not enough edges found

    SAME_THRESHOLD = 0.3
    sorted_x = sorted_x[np.diff(sorted_x, prepend=-median_size) > (SAME_THRESHOLD * median_size)]
    sorted_y = sorted_y[np.diff(sorted_y, prepend=-median_size) > (SAME_THRESHOLD * median_size)]

    median_diff = np.median(np.concatenate([np.diff(sorted_x), np.diff(sorted_y)]))

    CLOSE_THRESHOLD = 0.7
    FAR_THRESHOLD = 1.3
    min_size = CLOSE_THRESHOLD * median_diff
    max_size = FAR_THRESHOLD * median_diff
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

    median_diff = np.median(np.concatenate([np.diff(sorted_x), np.diff(sorted_y)]))

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

    return sorted_x, sorted_y, int(median_diff)


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

        # Find the minefield in the selection
        self.minefield_rect_in_selection = find_minefield(screen)
        (s_left, s_top), (s_right, s_bottom) = self.selected_rect
        (m_left, m_top), (m_right, m_bottom) = self.minefield_rect_in_selection
        self.minefield_rect_global = ((s_left + m_left, s_top + m_top),
                                      (s_left + m_right, s_top + m_bottom))

        screen = self.get_screen(self.minefield_rect_global)

        # Find the grid of tiles
        self.vertical_lines, self.horizontal_lines, self.tile_size = detect_tiles(screen)
        self.n_cols = len(self.vertical_lines)
        self.n_rows = len(self.horizontal_lines)
        self.is_grid_detected = self.tile_size != -1

    # Debug
    def draw_debug_grid(self, screen: Img):
        pass # TODO

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