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

LIGHT_GRAY = (198, 198, 198)


###################################################################
#                  MINESWEEPER STATE RECOGNITION                  #
###################################################################

def find_minefield(screen: Img) -> Rect:
    # processed = cv.inRange(screen, LIGHT_GRAY, LIGHT_GRAY)
    processed_screen = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    contours, _ = cv.findContours(processed_screen, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(screen, [cnt], 0, (0, 0, 255), 2)
    # cv.imshow('contours', screen)

    minefield_area = 0
    best_rect = ((0, 0), (0, 0))
    for cnt in contours:
        left, top, width, height = cv.boundingRect(cnt)
        rect_area = height * width
        if rect_area > minefield_area:
            minefield_area = rect_area
            best_rect = ((left, top), (left + width, left + height))

    return best_rect


###################################################################
#                               GUI                               #
###################################################################

WINDOW_NAME = 'Minesweeper Cheat'
MAX_FPS = 60


def setup():
    cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)


def main_loop():
    setup()

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

        # Find the minefield
        minefield_rect = find_minefield(screen)
        # print(minefield_rect)
        cv.rectangle(screen, minefield_rect[0], minefield_rect[1], (0, 255, 0), 2)

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
    main_loop()