import cv2 as cv
import numpy as np
import pyautogui
import typing as tp
from time import time_ns
from scipy.stats import mode # type: ignore
from enum import Enum


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


def detect_tiles(img: Img) -> tp.Tuple[tp.List[int], tp.List[int], int]:
    img = extract_gray(img)

    # Detect horizontal and vertical lines
    LINES_THRESHOLD = 50

    ## Detect vertical lines
    lighter_than_left = img[1:, 1:] > img[1:, :-1]
    same_as_up = img[1:, 1:] == img[:-1, 1:]
    left_non_black = img[1:, :-1] != 0
    vertical_mask = lighter_than_left & same_as_up & left_non_black
    vertical_lines = cv.HoughLines(vertical_mask.astype(np.uint8), 1, np.pi / 2, LINES_THRESHOLD)
    vertical_lines_x = []
    if vertical_lines is None:
        return [], [], -1 # No need to return horizontal lines, as we need both
    for line in vertical_lines:
        rho, theta = line[0]
        vertical_lines_x.append(rho)

    ## Detect horizontal lines
    lighter_than_up = img[1:, 1:] > img[:-1, 1:]
    same_as_left = img[1:, 1:] == img[1:, :-1]
    up_non_black = img[:-1, 1:] != 0
    horizontal_mask = lighter_than_up & same_as_left & up_non_black
    horizontal_lines = cv.HoughLines(horizontal_mask.astype(np.uint8), 1, np.pi / 2, 100)
    horizontal_lines_y = []
    if horizontal_lines is None:
        return [], [], -1 # No need to return vertical lines, as we need both
    for line in horizontal_lines:
        rho, theta = line[0]
        horizontal_lines_y.append(rho)

    # Merge lines that are close to each other into their average
    MIN_REL_DIFF = 0.3

    sorted_x = np.sort(vertical_lines_x)
    sorted_y = np.sort(horizontal_lines_y)
    if len(sorted_x) < 2 or len(sorted_y) < 2:
        return [], [], -1 # Not enough lines to detect tiles
    max_diff_x = int(np.max(np.diff(sorted_x)))

    ## Merge close vertical lines
    close_x_indices = np.flatnonzero(np.diff(sorted_x) < MIN_REL_DIFF * max_diff_x)
    if len(close_x_indices):
        first_close_x = close_x_indices[0]
        prev_close_x = close_x_indices[0]
        for i in close_x_indices:
            if i - prev_close_x > 1:
                sorted_x[first_close_x : prev_close_x + 2] \
                    = np.mean(sorted_x[first_close_x : prev_close_x + 2])
                first_close_x = i
            prev_close_x = i
        sorted_x[first_close_x : prev_close_x + 2] \
            = np.mean(sorted_x[first_close_x : prev_close_x + 2])
        sorted_x = np.unique(sorted_x)
        # // print(np.diff(sorted_x))

    ## Merge close horizontal lines
    close_y_indices = np.flatnonzero(np.diff(sorted_y) < MIN_REL_DIFF * max_diff_x)
    if len(close_y_indices):
        first_close_y = close_y_indices[0]
        prev_close_y = close_y_indices[0]
        for i in close_y_indices:
            if i - prev_close_y > 1:
                sorted_y[first_close_y : prev_close_y + 2] \
                    = np.mean(sorted_y[first_close_y : prev_close_y + 2])
                first_close_y = i
            prev_close_y = i
        sorted_y[first_close_y : prev_close_y + 2] \
            = np.mean(sorted_y[first_close_y : prev_close_y + 2])
        sorted_y = np.unique(sorted_y)
        # // print(np.diff(sorted_y))

    # Calculate the tile size
    all_diffs = np.concatenate((np.diff(sorted_x), np.diff(sorted_y)))
    tile_size = int(mode(all_diffs, keepdims=False)[0])

    # Debug
    if cv.waitKey(1) == ord('d'):
        dbg_img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for x in sorted_x:
            cv.line(dbg_img, (int(x), 0), (int(x), dbg_img.shape[0]), (0, 0, 255), 1)
        for y in sorted_y:
            cv.line(dbg_img, (0, int(y)), (dbg_img.shape[1], int(y)), (0, 0, 255), 1)
        cv.imshow('debug', dbg_img)
        cv.waitKey(0)

        dbg_img = cv.cvtColor((vertical_mask | horizontal_mask).astype(np.uint8) * 255, cv.COLOR_GRAY2BGR)
        # // for x in sorted_x:
        # //     cv.line(dbg_img, (int(x), 0), (int(x), dbg_img.shape[0]), (0, 0, 255), 1)
        # // for y in sorted_y:
        # //     cv.line(dbg_img, (0, int(y)), (dbg_img.shape[1], int(y)), (0, 0, 255), 1)
        cv.imshow('debug', dbg_img)
        cv.waitKey(0)
        cv.destroyWindow('debug')

    # Remove lines that are not evenly spaced
    TILE_SIZE_MAX_REL_DIFF = 0.25
    max_diff_diff = TILE_SIZE_MAX_REL_DIFF * tile_size
    while len(sorted_x) > 3:
        diff_diff_tile_size = np.abs(np.diff(sorted_x) - tile_size)
        good_x_lines_mask = np.zeros(len(sorted_x), dtype=bool)
        good_x_lines_mask[:-1] |= diff_diff_tile_size <= max_diff_diff
        good_x_lines_mask[1:] |= diff_diff_tile_size <= max_diff_diff
        if np.all(good_x_lines_mask) or np.all(~good_x_lines_mask):
            break
        # bad lines = lines with bad diff against a neighboring good line
        bad_x_lines_mask = np.zeros(len(sorted_x), dtype=bool)
        bad_x_lines_mask[:-1] |= (~good_x_lines_mask[:-1]) & good_x_lines_mask[1:]
        bad_x_lines_mask[1:] |= (~good_x_lines_mask[1:]) & good_x_lines_mask[:-1]
        # remove bad lines
        print('x', sorted_x, np.diff(sorted_x), good_x_lines_mask, bad_x_lines_mask) # debug
        sorted_x = sorted_x[~bad_x_lines_mask]
    while len(sorted_y) > 3:
        diff_diff_tile_size = np.abs(np.diff(sorted_y) - tile_size)
        good_y_lines_mask = np.zeros(len(sorted_y), dtype=bool)
        good_y_lines_mask[:-1] |= diff_diff_tile_size <= max_diff_diff
        good_y_lines_mask[1:] |= diff_diff_tile_size <= max_diff_diff
        if np.all(good_y_lines_mask) or np.all(~good_y_lines_mask):
            break
        # bad lines = lines with bad diff against a neighboring good line
        bad_y_lines_mask = np.zeros(len(sorted_y), dtype=bool)
        bad_y_lines_mask[:-1] |= (~good_y_lines_mask[:-1]) & good_y_lines_mask[1:]
        bad_y_lines_mask[1:] |= (~good_y_lines_mask[1:]) & good_y_lines_mask[:-1]
        # remove bad lines
        print('y', sorted_y, np.diff(sorted_y), good_y_lines_mask, bad_y_lines_mask) # debug
        sorted_y = sorted_y[~bad_y_lines_mask]

    return [int(x) for x in sorted_x], [int(y) for y in sorted_y], tile_size


###################################################################
#                        MAIN CLASS AND GUI                       #
###################################################################

class Minesweeper:
    def __init__(self):
        self.setup()
        self.main_loop()

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

    def setup(self):
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
            Finds the minefield in the selection.
        """

        (left, top), (right, bottom) = self.selected_rect
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        self.selected_rect = ((left, top), (right, bottom))

        screen: Img = self.get_screen(self.selected_rect)
        self.minefield_rect_in_selection = find_minefield(screen)
        (s_left, s_top), (s_right, s_bottom) = self.selected_rect
        (m_left, m_top), (m_right, m_bottom) = self.minefield_rect_in_selection
        self.minefield_rect_global = ((s_left + m_left, s_top + m_top),
                                      (s_left + m_right, s_top + m_bottom))

    def main_loop(self):
        """ The main loop of the program. """

        target_delta_time = 1e9 / MAX_FPS
        prev_time = time_ns()
        while True:
            if self.selected_rect == NONE_RECT:
                screen = self.get_screen(mask_window=True)
            else:
                screen = self.get_screen(self.minefield_rect_global)

            # Detect game state

            # Detect the grid and tiles
            # TODO: Don't do this every frame, just in handle_selected_rect_change, THIS IS JUST FOR TESTING
            vertical_lines, horizontal_lines, tile_size = detect_tiles(screen)

            # debug
            screen = extract_gray(screen)
            # Set all non-masked pixels to last masked color left of them
            for x in vertical_lines:
                cv.line(screen, (x, 0), (x, screen.shape[0]), (0, 0, 255), 1)
            for y in horizontal_lines:
                cv.line(screen, (0, y), (screen.shape[1], y), (0, 0, 255), 1)
            cv.rectangle(screen, (0, 0), (tile_size, tile_size), (255, 0, 0), 2)

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
    Minesweeper()