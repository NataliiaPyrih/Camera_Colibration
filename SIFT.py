import cv2
import numpy as np
import yaml

sift = cv2.SIFT_create()
images=["clog.jpg", "clog2.jpg"]

def get_key_points(img):
    image = cv2.imread(img)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = sift.detectAndCompute(image_gray, None)
    image_keypoints=cv2.drawKeypoints(image_gray, keypoints, image, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_name=img.split('.')[0] + '_with_keypoints.jpg'
    cv2.imwrite(image_name, image_keypoints)
    return image, keypoints, descriptors


img1, kp1, desc1=get_key_points(images[0])
img2, kp2, desc2=get_key_points(images[1])

flann = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_FLANNBASED)
matches = flann.knnMatch(desc1, desc2,k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches)>10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    warped_image = cv2.warpPerspective(img1, homography, (img2.shape[1], img2.shape[0]))

    result_image = cv2.addWeighted(warped_image, 0.5, img2, 0.5, 0)
    cv2.imwrite("homography_image.jpg",result_image)
    cv2.imshow('Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("There are not enough good comparisons to evaluate homography")

def get_known_camera_matrix():
    with open('calibration_matrix.yaml', 'r') as file:
        data = yaml.safe_load(file)
    camera_matrix = data['camera_matrix']
    dist_coeff = data['dist_coeff']
    return camera_matrix, dist_coeff

known_camera_matrix, known_dist_coeff = get_known_camera_matrix()
P2 = cv2.getPerspectiveTransform(known_camera_matrix, homography)
camera_matrix2, dist_coeffs2, R, t = cv2.decomposeProjectionMatrix(P2)

