import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob


def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    scaled_mag = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
    return mag_binary


def calibrate_camera():
    imgs = [cv2.imread(filename) for filename in glob.glob("camera_cal/*.jpg")]
    nx = 9
    ny = 6
    objpoints = []
    imgpoints = []
    # Construct object points
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    for img in imgs:
        ret, corners = cv2.findChessboardCorners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (nx, ny))
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, imgs[0].shape[1::-1], None, None)
    return ret, mtx, dist


def undistort(image):
    ret, mtx, dist = calibrate_camera()
    return cv2.undistort(image, mtx, dist, None, mtx)


def warp_to_bird_eye(img):
    src = np.float32(
        [[571, 468],
         [244.661, 693.952],
         [1042.08, 681.048],
         [716, 468]])
    dst = np.float32(
        [[img.shape[1] / 4, 0],
         [img.shape[1] / 4, img.shape[0]],
         [img.shape[1] * 3 / 4, img.shape[0]],
         [img.shape[1] * 3 / 4, 0]])
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (img.shape[1], img.shape[0]), flags=cv2.INTER_NEAREST), m


def s_channel(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]


def color_threshold(image, thresh=(0, 255)):
    binary = np.zeros_like(image)
    binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return binary


def find_lane_pixels(binary):
    nwindows = 9
    margin = 100
    minpix = 50
    bottom_half = binary[binary.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    mid_point = histogram.shape[0] // 2
    left_x_current, right_x_current = np.argmax(histogram[0:mid_point]), np.argmax(histogram[mid_point:]) + mid_point
    out_img = np.dstack((binary, binary, binary))
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    window_height = binary.shape[0] // nwindows
    left_inds = []
    right_inds = []
    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = win_y_low + window_height
        left_win_x_low, left_win_x_high = left_x_current - margin, left_x_current + margin
        right_win_x_low, right_win_x_high = right_x_current - margin, right_x_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (left_win_x_low, win_y_low),
                      (left_win_x_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (right_win_x_low, win_y_low),
                      (right_win_x_high, win_y_high), (0, 255, 0), 2)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                          & (nonzerox >= left_win_x_low) & (nonzerox <= left_win_x_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                           & (nonzerox >= right_win_x_low) & (nonzerox <= right_win_x_high)).nonzero()[0]
        left_inds.append(good_left_inds)
        right_inds.append(good_right_inds)
        left_x_inds = nonzerox[good_left_inds]
        right_x_inds = nonzerox[good_right_inds]
        if len(left_x_inds) > minpix:
            left_x_current = np.int(np.mean(left_x_inds))
        if len(right_x_inds) > minpix:
            right_x_current = np.int(np.mean(right_x_inds))
    left_lane_inds = np.concatenate(left_inds)
    right_lane_inds = np.concatenate(right_inds)
    return nonzerox[left_lane_inds], nonzeroy[left_lane_inds],\
        nonzerox[right_lane_inds], nonzeroy[right_lane_inds], out_img


def fit_poly(binary):
    left_x, left_y, right_x, right_y, out_img = find_lane_pixels(binary)
    left_fit = np.polyfit(left_y, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)
    plot_y = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    left_fit_x = left_fit[0] * plot_y ** 2 + left_fit[1] * plot_y + left_fit[2]
    right_fit_x = right_fit[0] * plot_y ** 2 + right_fit[1] * plot_y + right_fit[2]
    # Visualization
    # Colors in the left and right lane regions
    out_img[left_y, left_x] = [255, 0, 0]
    out_img[right_y, right_x] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fit_x, plot_y, color='yellow')
    # plt.plot(right_fit_x, plot_y, color='yellow')

    return left_fit_x, right_fit_x, plot_y, out_img


undist = undistort(cv2.imread("test_images/test6.jpg"))
s = s_channel(undist)
mag_binary = mag_threshold(s, sobel_kernel=5, mag_thresh=(30, 250))
color_binary = color_threshold(s, thresh=(80, 255))
rgb_binary = color_threshold(cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY), thresh=(40, 255))
combined = np.zeros_like(mag_binary)
combined[(mag_binary == 1) & (color_binary == 1) & (rgb_binary == 1)] = 1
warped, m = warp_to_bird_eye(combined)
# Create an image to draw the lines on
warp_zero = np.zeros_like(warped).astype(np.uint8)
color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
left_fit_x, right_fit_x, plot_y, out_img = fit_poly(warped)

# Recast the x and y points into usable format for cv2.fillPoly()
pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
pts = np.hstack((pts_left, pts_right))

# Draw the lane onto the warped blank image
cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

# Warp the blank back to original image space using inverse perspective matrix (Minv)
new_warp = cv2.warpPerspective(color_warp, np.linalg.inv(m), (warped.shape[1], warped.shape[0]))
# Combine the result with the original image
result = cv2.addWeighted(undist, 1, new_warp, 0.3, 0)
# plt.imshow(warped, cmap='gray')
# plt.imshow(out_img)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
plt.show()
