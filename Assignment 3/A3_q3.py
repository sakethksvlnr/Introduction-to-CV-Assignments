import cv2
import numpy as np

cap = cv2.VideoCapture('files/10sec_video.mp4')
print(cap.read()[1])
frames_dict = {}
count = 0
while cap.read()[0]:
    frames_dict[count] = cv2.resize(cap.read()[1], (640, 480))
    count += 1
print(frames_dict)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
feature_params = dict(maxCorners=100, qualityLevel=0.6, minDistance=5, blockSize=7)
frame_idx = 0
track_len = 4
tracks = []
flag = True

# Change this to 1, 11, 31
esc = 1
counter = 0
# if flag:
#     prev_frame = frame
#     prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     flag = False
for key, value in frames_dict.items():
    frame = frames_dict[counter + esc]
    prev_frame = frames_dict[counter]

    visualize = frame.copy()

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(frame_gray)
    mask[:] = 255

    # Detect the corner features in the image, based on the parameters discussed earlier.
    for x, y in [np.int32(tr[-1]) for tr in tracks]:
        cv2.circle(mask, (x, y), 5, 0, -1)
    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)

    # Once the good features are selected, append them to the "tracks" variable.
    if p is not None:
        for x, y in np.float32(p).reshape(-1, 2):
            tracks.append([(x, y)])

    if len(tracks) > 0:
        img0, img1 = prev_gray, frame_gray
        # Compute the optical flow of the good feature points using Lucas-Kanade method.
        # Also, compute the distance between the good feature points in the current and previous frame.
        p0 = np.float32([tr[-1] for tr in tracks])
        p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        # print len(d)

        # Append only those optical points, whose distance is less than 1.
        good = d < 1
        # print good
        new_tracks = []
        for tr, (x, y), good_flag in zip(tracks, p1.reshape(-1, 2), good):
            if not good_flag:
                continue
            tr.append((x, y))
            if len(tr) > track_len:
                del tr[0]
            new_tracks.append(tr)

        tracks = new_tracks
        cv2.polylines(visualize, [np.int32(tr) for tr in tracks], False, (200, 100, 0))

    frame_idx += 1
    counter+=1
    cv2.imshow("Optical Flow", visualize)

    if cv2.waitKey(30) == 27 or (counter + esc) >= 149:
        cv2.destroyAllWindows()
        break
