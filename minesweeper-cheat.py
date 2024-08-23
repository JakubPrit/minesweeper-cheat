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

SCREEN_PADDING = 10


###################################################################
#                  MINESWEEPER STATE RECOGNITION                  #
###################################################################

def find_minefield(screen: Img) -> Rect:
    padded_screen = screen.copy()
    padded_screen.fill(0)
    screen_height, screen_width = screen.shape[:2]
    left, top = SCREEN_PADDING, SCREEN_PADDING
    right, bottom = screen_width - SCREEN_PADDING, screen_height - SCREEN_PADDING
    padded_screen[top:bottom, left:right] = screen[top:bottom, left:right]
    edges = cv.Canny(padded_screen, 100, 200)
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(screen, contours, -1, (0, 0, 255), 2)
    # cv.imshow('contours', screen)

    minefield_area = 0
    best_rect = ((0, 0), (0, 0))
    largest_sizes = [0] * 10
    largest_rects = [((0, 0), (0, 0))] * 10
    for cnt in contours:
        left, top, width, height = cv.boundingRect(cnt)
        rect_area = height * width
        for i in range(10):
            if rect_area > largest_sizes[i]:
                largest_sizes.insert(i, rect_area)
                largest_sizes.pop()
                largest_rects.insert(i, ((left, top), (left + width, top + height)))
                largest_rects.pop()
                break
        if rect_area > minefield_area:
            minefield_area = rect_area
            best_rect = ((left, top), (left + width, left + height))
    for rect in largest_rects:
        cv.rectangle(screen, rect[0], rect[1], (255, 0, 0), 1)

    return best_rect


###################################################################
#                        MAIN CLASS AND GUI                       #
###################################################################

WINDOW_NAME = 'Minesweeper Cheat'
MAX_FPS = 60

class Minesweeper:
    def __init__(self):
        self.setup()
        self.main_loop()

    def get_screen(self) -> Img:
        return cv.cvtColor(np.array(pyautogui.screenshot()), cv.COLOR_RGB2BGR)

    def setup(self):
        cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
        screen_width, screen_height = pyautogui.size()
        self.selected_rect: Rect = ((0, 0), (screen_width, screen_height))
        cv.setMouseCallback(WINDOW_NAME, self.handle_mouse_event) # type: ignore

    def handle_mouse_event(self, event, x, y, flags, param):
        if event in (cv.EVENT_LBUTTONDOWN, cv.EVENT_LBUTTONUP):
            screen_width, screen_height = pyautogui.size()
            window_width, window_height = cv.getWindowImageRect(WINDOW_NAME)[2:4]
            rel_x, rel_y = x / window_width, y / window_height
            print(rel_x, rel_y, x, y, window_width, window_height)
            screen_x, screen_y = int(screen_width * rel_x), int(screen_height * rel_y)
        if event == cv.EVENT_LBUTTONDOWN:
            self.selected_rect = ((screen_x, screen_y), self.selected_rect[1])
        elif event == cv.EVENT_LBUTTONUP:
            self.selected_rect = (self.selected_rect[0], (screen_x, screen_y))
            self.handle_selected_rect_change()

    def handle_selected_rect_change(self):
        (left, top), (right, bottom) = self.selected_rect
        if left > right:
            left, right = right, left
        if top > bottom:
            top, bottom = bottom, top
        self.selected_rect = ((left, top), (right, bottom))

        screen: Img = self.get_screen()
        selected_screen_part = screen[top:bottom, left:right]

        # Find the minefield
        minefield_rect = find_minefield(selected_screen_part)
        cv.rectangle(selected_screen_part, minefield_rect[0], minefield_rect[1], (0, 255, 0), 2)
        cv.imshow(WINDOW_NAME, selected_screen_part)
        cv.waitKey(5000)


    def main_loop(self):
        target_delta_time = 1e9 / MAX_FPS
        prev_time = time_ns()
        while True:
            # Get screen content
            screen: Img = cv.cvtColor(np.array(pyautogui.screenshot()), cv.COLOR_RGB2BGR)

            # Cut out (fill with black) the window of this program
            win_pos = cv.getWindowImageRect(WINDOW_NAME)
            left, top, width, height = win_pos
            right, bottom = left + width, top + height
            if width > 0 and height > 0:
                screen[top:bottom, left:right] = 0

            # Show the screen
            cv.imshow(WINDOW_NAME, screen)

            # Limit the frame rate
            curr_time = time_ns()
            delta_time = curr_time - prev_time
            prev_time = curr_time
            wait_time = max(1, int(target_delta_time - delta_time) // 10**6)
            if cv.waitKey(wait_time) == ord('q'):
                break # Quit if 'q' is pressed

        cv.destroyAllWindows()


if __name__ == '__main__':
    Minesweeper()