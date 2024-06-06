import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
from time import sleep

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

def draw_button(frame, button, is_finger_on_button, is_landmark_8_over_it, color):
    x, y = button.pos
    w, h = button.size

    # Darken the color if only landmark 8 is over the button
    if is_landmark_8_over_it and not is_finger_on_button:
        color = tuple(c * 0.5 for c in color)

    # Create a transparent rectangle
    overlay = frame.copy()
    cv.rectangle(overlay, button.pos, (x + w, y + h), color, cv.FILLED)
    cv.putText(overlay, button.text, (x + 15, y + 70), cv.FONT_HERSHEY_PLAIN, 5, (5, 5, 5), 6)

    # Blend the overlay with the original frame
    alpha = 0.5  # You can adjust the transparency level here
    cv.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    return frame

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1200)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)


detector = HandDetector(detectionCon=0.8)

keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
]

finalText = ""

button_list = []

# Creating rows
for i, row in enumerate(keys):
    for j, key in enumerate(row):
        button_list.append(Button([100 * j + 50, 100 * i + 50], key))

while True:
    success, frame = cap.read()
    hands, frame = detector.findHands(frame)

    if hands:
        landmarks = hands[0]['lmList']
        print(landmarks)

        for button in button_list:
            x, y = button.pos
            w, h = button.size

            # Check if the finger is on the button
            is_finger_on_button = x < landmarks[8][0] < x + w and y < landmarks[8][1] < y + h

            # Check if only landmark 8 is over the button
            is_landmark_8_over_it = x < landmarks[8][0] < x + w and y < landmarks[8][1] < y + h and \
                                    not (x < landmarks[12][0] < x + w and y < landmarks[12][1] < y + h)

            # Get the distance between landmarks 12 and 8
            length, _, _ = detector.findDistance(landmarks[8][:2], landmarks[12][:2], frame)
            print(length)

            if length < 30 and is_finger_on_button:
                color = (100, 120, 100)
                finalText += button.text  # Set the finalText to the button text when length < 30
                sleep(.30)
            else:
                color = (0, 120, 0) if is_finger_on_button else (250, 120, 120)

            frame = draw_button(frame, button, is_finger_on_button, is_landmark_8_over_it, color)

    cv.rectangle(frame, (40, 350), (600, 450), (175, 0, 175), cv.FILLED)
    cv.putText(frame, finalText, (50, 400), cv.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    cv.imshow("Webcam detected", frame)
    if cv.waitKey(1) == 27:  # Press 'Esc' to exit
        break

cap.release()
cv.destroyAllWindows()
