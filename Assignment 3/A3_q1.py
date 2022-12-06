import random
import numpy as np
import cv2


def visualize_image(frame_name, frame):
    cv2.imshow(frame_name, frame)
    cv2.imwrite(frame_name + ".jpg", frame)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cap = cv2.VideoCapture("files/10sec_video.mp4")
    obj_image = cv2.imread("files/object.png")
    frames_dict = {}
    count = 0

    while cap.read()[0]:
        frames_dict[count] = cap.read()[1]
        count += 1

    count = 0
    print(frames_dict)
    for i in range(10):
        # loading a random image
        random_frame = frames_dict[random.choice([x for x in range(80, 120)])]
        gray = cv2.cvtColor(random_frame, cv2.COLOR_BGR2GRAY)
        gray_obj = cv2.cvtColor(obj_image, cv2.COLOR_BGR2GRAY)

        # Create Summed of Squared Differences with patch
        ssd = cv2.matchTemplate(gray, gray_obj, cv2.TM_SQDIFF_NORMED)

        # find min for match with SQDIFF_NORMED
        point = np.where(ssd == ssd.min())
        y = point[0][0]
        x = point[1][0]
        w = len(obj_image[0])
        l = len(obj_image)

        # Draw Rectangle
        cv2.rectangle(random_frame, (x, y), (x + w, y + l), (0, 255, 0), 3)

        visualize_image("test_frame_{}".format(count), random_frame)
        count+=1
