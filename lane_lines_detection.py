import cv2
import numpy as np
import glob
from moviepy.editor import VideoFileClip

# The left and right polyline fit
left_fit, right_fit = None, None
# We maintain the last 5 frames to compute the average polyline fit
max_averge_frame = 5
# The left and right polyline fit in the last 5 frames are kept in this list
previous_left_fits, previous_right_fits = [], []
lost_frame = 0
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


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


def warp_to_bird_eye(img):
    """Apply the perspective transformation to warp the images to the bird eye view"""
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


def mag_threshold(image, sobel_kernel=3, mag_thresh=(0, 255)):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    mag_sobel = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    scaled_mag = np.uint8(255 * mag_sobel / np.max(mag_sobel))
    mag_binary = np.zeros_like(scaled_mag)
    mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1
    return mag_binary


def color_threshold(image, thresh=(0, 255)):
    binary = np.zeros_like(image)
    binary[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return binary


def find_lane_pixels(binary):
    nwindows = 9
    margin = 50
    minpix = 50
    bottom_half = binary[binary.shape[0] // 2:, :]
    histogram = np.sum(bottom_half, axis=0)
    mid_point = histogram.shape[0] // 2
    start = 100
    left_x_current, right_x_current =\
        np.argmax(histogram[start:mid_point]) + start, np.argmax(histogram[mid_point:-start]) + mid_point
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
    return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], nonzerox[right_lane_inds], nonzeroy[right_lane_inds]


def find_lane_pixels_prior(binary, left_fit, right_fit):
    """Find the lane pixels based on the previous polyline fit"""
    margin = 50
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                      left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                       left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                       right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
    return nonzerox[left_lane_inds], nonzeroy[left_lane_inds], nonzerox[right_lane_inds], nonzeroy[right_lane_inds]


def curvature(y, fit):
    return (1 + (2 * fit[0] * y + fit[1]) ** 2) ** 1.5 / (2 * abs(fit[0]))


def position(y, fit):
    return fit[0] * y ** 2 + fit[1] * y + fit[2]


def fit_poly(binary):
    """Calculate the left and right polylines that fit the lane pixels"""
    global left_fit, right_fit, lost_frame
    # If lane line finding is failed for 5 consecutive frame, start over
    if left_fit is None or lost_frame >= 5:
        left_x, left_y, right_x, right_y = find_lane_pixels(binary)
    else:
        left_x, left_y, right_x, right_y = find_lane_pixels_prior(binary, left_fit, right_fit)
    if len(left_x) < 5 or len(right_x) < 5:
        raise ValueError("Lane lines are not found")
    left_fit_current = np.polyfit(left_y, left_x, 2)
    right_fit_current = np.polyfit(right_y, right_x, 2)
    plot_y = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    y = np.max(plot_y)
    left_curvature = curvature(y, left_fit_current)
    right_curvature = curvature(y, right_fit_current)
    if 0.2 < left_curvature / right_curvature < 5 or left_fit is None:
        # The left curvature and right curvature are similar, update the polyline, otherwise
        # use the previous detected polyline
        if len(previous_left_fits) >= max_averge_frame:
            previous_left_fits.pop(0)
        if len(previous_right_fits) >= max_averge_frame:
            previous_right_fits.pop(0)
        previous_left_fits.append(left_fit_current)
        previous_right_fits.append(right_fit_current)
        left_fit = np.mean(np.array(previous_left_fits), axis=0)
        right_fit = np.mean(np.array(previous_right_fits), axis=0)
        lost_frame = 0
    else:
        lost_frame += 1
    left_fit_x = position(plot_y, left_fit)
    right_fit_x = position(plot_y, right_fit)
    left_fit_in_meter = np.polyfit(plot_y * ym_per_pix, left_fit_x * xm_per_pix, 2)
    right_fit_in_meter = np.polyfit(plot_y * ym_per_pix, right_fit_x * xm_per_pix, 2)
    left_curvature_in_meter = curvature(y * ym_per_pix, left_fit_in_meter)
    right_curvature_in_meter = curvature(y * ym_per_pix, right_fit_in_meter)
    lane_center = (position(y, left_fit) + position(y, right_fit)) / 2
    offset = xm_per_pix * abs(binary.shape[1] / 2 - lane_center)
    return left_fit_x, right_fit_x, plot_y, left_curvature_in_meter, right_curvature_in_meter, offset


def draw_text(image, text, bottom_left):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 255, 255)
    line_type = 2
    cv2.putText(image, text, bottom_left, font, font_scale, font_color, line_type)
    return image


def process_image(image):
    s = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mag_gray_binary = mag_threshold(gray, sobel_kernel=5, mag_thresh=(50, 250))
    mag_s_binary = mag_threshold(s, sobel_kernel=5, mag_thresh=(20, 250))
    color_binary = color_threshold(gray, thresh=(40, 255))
    combined = np.zeros_like(gray)
    # Calculate the binary images by the combination on the sobel magnitude filter on grayscale and s channel
    # and the color thresholding on the grayscale image
    combined[((mag_s_binary == 1) | (mag_gray_binary == 1)) & (color_binary == 1)] = 1
    warped, m = warp_to_bird_eye(combined)
    try:
        left_fit_x, right_fit_x, plot_y, left_curvature, right_curvature, offset = fit_poly(warped)
    except ValueError:
        return image

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Calculate the curvature
    y = np.max(plot_y)
    # Draw left and right curvature on the images
    draw_text(image, 'left curvature ' + str(left_curvature), (400, 100))
    draw_text(image, 'right curvature ' + str(right_curvature), (400, 150))
    draw_text(image, 'offset to the lane center ' + str(offset), (400, 200))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    new_warp = cv2.warpPerspective(color_warp, np.linalg.inv(m), (warped.shape[1], warped.shape[0]))
    # Combine the result with the original image
    return cv2.addWeighted(image, 1, new_warp, 0.3, 0)


ret, mtx, dist = calibrate_camera()
clip = VideoFileClip("project_video.mp4")
white_clip = clip.fl_image(lambda img: process_image(cv2.undistort(img, mtx, dist, None, mtx)))
white_clip.write_videofile("output_images/output.mp4", audio=False)
