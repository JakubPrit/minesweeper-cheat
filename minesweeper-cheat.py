import cv2 as cv
import numpy as np
import pyautogui
import typing as tp
from time import time_ns


###################################################################
#                  TYPE DEFINITIONS AND CONSTANTS                 #
###################################################################

Pos = tp.Tuple[int, int]
Rect = tp.Tuple[Pos, Pos]
Img = np.ndarray


WINDOW_NAME = 'Minesweeper Cheat'
MAX_FPS = 60


###################################################################
#                  MINESWEEPER STATE RECOGNITION                  #
###################################################################

def find_minefield(screen: Img) -> Rect:
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


def get_gray_mask(img: Img, threshold: int = 10) -> Img:
    red_blue_diff = np.abs(img[:, :, 2].astype(np.int16) - img[:, :, 0].astype(np.int16))
    red_green_diff = np.abs(img[:, :, 2].astype(np.int16) - img[:, :, 1].astype(np.int16))
    blue_green_diff = np.abs(img[:, :, 0].astype(np.int16) - img[:, :, 1].astype(np.int16))
    return (red_blue_diff < threshold) & (red_green_diff < threshold) & (blue_green_diff < threshold)


def detect_tiles(img: Img) -> tp.List[Rect]:
    mask = get_gray_mask(img).astype(np.uint8) * 255
    img = cv.cvtColor(cv.bitwise_and(img, img, mask=mask), cv.COLOR_BGR2GRAY)
    debug_img = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

    lighter_than_left = img[:, 1:] > img[:, :-1]
    same_as_up = img[1:, :] == img[:-1, :]
    vertical_mask = lighter_than_left[1:, :] & same_as_up[:, 1:]
    vertical_lines = cv.HoughLines(vertical_mask.astype(np.uint8), 1, np.pi / 2, 100)
    vertical_lines_x = []
    for line in vertical_lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv.line(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        vertical_lines_x.append(rho)

    lighter_than_up = img[1:, :] > img[:-1, :]
    same_as_left = img[:, 1:] == img[:, :-1]
    horizontal_mask = lighter_than_up[:, 1:] & same_as_left[1:, :]
    horizontal_lines = cv.HoughLines(horizontal_mask.astype(np.uint8), 1, np.pi / 2, 100)
    horizontal_lines_y = []
    for line in horizontal_lines:
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * a)
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * a)
        cv.line(debug_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        horizontal_lines_y.append(rho)

    for x in vertical_lines_x:
        cv.line(debug_img, (int(x), 0), (int(x), img.shape[0]), (0, 0, 255), 1)
    for y in horizontal_lines_y:
        cv.line(debug_img, (0, int(y)), (img.shape[1], int(y)), (0, 0, 255), 1)

    tile_mask = vertical_mask | horizontal_mask
    cv.imshow('Tile mask', tile_mask.astype(np.uint8) * 255)
    cv.waitKey(0)
    cv.imshow('Debug', debug_img)
    cv.waitKey(0)

    return []


###################################################################
#                        MAIN CLASS AND GUI                       #
###################################################################

class Minesweeper:
    def __init__(self):
        self.minefield_rect: Rect = ((0, 0), (0, 0))
        self.setup()
        self.main_loop()

    def get_screen(self) -> Img:
        return cv.cvtColor(np.array(pyautogui.screenshot()), cv.COLOR_RGB2BGR)

    def reset_selection(self):
        screen_width, screen_height = pyautogui.size()
        self.selection_corner: Pos = (0, 0)
        self.selected_rect: Rect = ((0, 0), (screen_width, screen_height))

    def setup(self):
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        self.reset_selection()
        cv.setMouseCallback(WINDOW_NAME, self.handle_mouse_event) # type: ignore

    def handle_mouse_event(self, event, *_):
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
        (left, top), (right, bottom) = self.selected_rect
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        self.selected_rect = ((left, top), (right, bottom))

        # screen: Img = self.get_screen()
        # selected_screen_part = screen[top:bottom, left:right]
        # self.minefield_rect = find_minefield(selected_screen_part)

    def main_loop(self):
        target_delta_time = 1e9 / MAX_FPS
        prev_time = time_ns()
        while True:
            # Get screen content
            full_screen: Img = cv.cvtColor(np.array(pyautogui.screenshot()), cv.COLOR_RGB2BGR)

            # Cut out (fill with black) the window of this program
            win_pos = cv.getWindowImageRect(WINDOW_NAME)
            left, top, width, height = win_pos
            right, bottom = left + width, top + height
            if width > 0 and height > 0:
                full_screen[top:bottom, left:right] = 0

            # Crop to the selected area
            (left, top), (right, bottom) = self.selected_rect
            screen = full_screen[top:bottom, left:right]

            # Find the minefield # TODO: Don't do this every frame, just in handle_selected_rect_change, THIS IS JUST FOR TESTING
            self.minefield_rect = find_minefield(screen)
            cv.rectangle(screen, self.minefield_rect[0], self.minefield_rect[1], (0, 255, 0), 2)

            # Show the screen
            detect_tiles(screen)
            mask = get_gray_mask(screen).astype(np.uint8) * 255
            screen = cv.cvtColor(cv.bitwise_and(screen, screen, mask=mask), cv.COLOR_BGR2GRAY)
            cv.imshow(WINDOW_NAME, screen)

            # Limit the frame rate
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