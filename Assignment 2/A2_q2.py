import cv2

test_images = {'test1': ['files/dahlberg1.jpg', 'files/dahlberg2.jpg', 'files/dahlberg3.jpg'],
               'test2': ['files/studentcentereast1.jpg', 'files/studentcentereast2.jpg', 'files/studentcentereast3.jpg'],
               'test3': ['files/urbanlife1.jpg', 'files/urbanlife2.jpg', 'files/urbanlife3.jpg'],
               'test4':['files/sciencecenter1.jpg','files/sciencecenter2.jpg','files/sciencecenter3.jpg'],
               'test5':['files/classroom_south1.jpg','files/classroom_south2.jpg','files/classroom_south3.jpg']}
for key, value in test_images.items():
    images = []
    print(key)
    for i in range(len(value)):
        images.append(cv2.imread(value[i]))
        images[i] = cv2.resize(images[i], (0, 0), fx=0.4, fy=0.4)
    image_sticher = cv2.Stitcher.create()
    (dummy, output) = image_sticher.stitch(images)

    cv2.imshow('Stitched Image', output)
    cv2.imwrite(key+'.jpg', output)
    cv2.waitKey(1)
