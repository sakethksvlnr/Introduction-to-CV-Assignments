import cv2 as cv
import numpy as np


def extract_features_orb_and_stitch_images(frame1, frame2):
    gray_1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray_2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    algortihm = cv.ORB_create()

    kp_img1, desc_img1 = algortihm.detectAndCompute(gray_1, None)
    kp_img2, desc_img2 = algortihm.detectAndCompute(gray_2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc_img2, desc_img1, k=2)

    features = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            features.append(m)

    query_pts = np.float32([kp_img2[m.queryIdx]
                           .pt for m in features]).reshape(-1, 1, 2)
    train_pts = np.float32([kp_img1[m.trainIdx]
                           .pt for m in features]).reshape(-1, 1, 2)
    print(query_pts)
    print(train_pts)
    matrix, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
    dst = cv.warpPerspective(frame2, matrix, ((frame1.shape[1] + frame2.shape[1]), frame2.shape[0]))
    dst[0:frame1.shape[0], 0:frame1.shape[1]] = frame1

    return dst


def visualize_image(frame_name, frame):
    cv.imshow(frame_name, frame)
    cv.imwrite(frame_name + ".jpg", frame)
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()


if __name__ == "__main__":
    frame1 = cv.imread('files/urbanlife1.jpg')
    visualize_image("Frame1", frame1)

    frame2 = cv.imread('files/urbanlife2.jpg')
    visualize_image("Frame2", frame2)

    stitched_image = extract_features_orb_and_stitch_images(frame1, frame2)

    visualize_image("Stitched_Image_ORB", stitched_image)
