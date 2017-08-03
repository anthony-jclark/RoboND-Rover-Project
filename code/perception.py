import numpy as np
import cv2

# Identify pixels above the threshold
# Threshold of RGB > 160 does a nice job of identifying ground pixels only
def color_thresh(img, rgb_thresh_min=(160, 160, 160), rgb_thresh_max=(255, 255, 255)):
    # Create an array of zeros same xy size as img, but single channel
    color_select = np.zeros_like(img[:,:,0])
    # Require that each pixel be above all three threshold values in RGB
    # above_thresh will now contain a boolean array with "True"
    # where threshold was met
    thresh_select = (rgb_thresh_min[0] <= img[:,:,0]) & (img[:,:,0] <= rgb_thresh_max[0]) \
                  & (rgb_thresh_min[1] <= img[:,:,1]) & (img[:,:,1] <= rgb_thresh_max[1]) \
                  & (rgb_thresh_min[2] <= img[:,:,2]) & (img[:,:,2] <= rgb_thresh_max[2])

            # Index the array of zeros with the boolean array and set to 1
    color_select[thresh_select] = 1
    # Return the binary image
    return color_select

# Define a function to convert from image coords to rover coords
def rover_coords(binary_img):
    # Identify nonzero pixels
    ypos, xpos = binary_img.nonzero()
    # Calculate pixel positions with reference to the rover position being at the
    # center bottom of the image.
    x_pixel = -(ypos - binary_img.shape[0]).astype(np.float)
    y_pixel = -(xpos - binary_img.shape[1]/2 ).astype(np.float)
    return x_pixel, y_pixel


# Define a function to convert to radial coords in rover space
def to_polar_coords(x_pixel, y_pixel):
    # Convert (x_pixel, y_pixel) to (distance, angle)
    # in polar coordinates in rover space
    # Calculate distance to each pixel
    dist = np.sqrt(x_pixel**2 + y_pixel**2)
    # Calculate angle away from vertical for each pixel
    angles = np.arctan2(y_pixel, x_pixel)
    return dist, angles

# Define a function to map rover space pixels to world space
def rotate_pix(xpix, ypix, yaw):
    # Convert yaw to radians
    yaw_rad = yaw * np.pi / 180
    xpix_rotated = (xpix * np.cos(yaw_rad)) - (ypix * np.sin(yaw_rad))

    ypix_rotated = (xpix * np.sin(yaw_rad)) + (ypix * np.cos(yaw_rad))
    # Return the result
    return xpix_rotated, ypix_rotated

def translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale):
    # Apply a scaling and a translation
    xpix_translated = (xpix_rot / scale) + xpos
    ypix_translated = (ypix_rot / scale) + ypos
    # Return the result
    return xpix_translated, ypix_translated


# Define a function to apply rotation and translation (and clipping)
# Once you define the two functions above this function should work
def pix_to_world(xpix, ypix, xpos, ypos, yaw, world_size, scale):
    # Apply rotation
    xpix_rot, ypix_rot = rotate_pix(xpix, ypix, yaw)
    # Apply translation
    xpix_tran, ypix_tran = translate_pix(xpix_rot, ypix_rot, xpos, ypos, scale)
    # Perform rotation, translation and clipping all at once
    x_pix_world = np.clip(np.int_(xpix_tran), 0, world_size - 1)
    y_pix_world = np.clip(np.int_(ypix_tran), 0, world_size - 1)
    # Return the result
    return x_pix_world, y_pix_world

# Define a function to perform a perspective transform
def perspect_transform(img, src, dst):

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))# keep same size as input image

    return warped


# Apply the above functions in succession and update the Rover state accordingly
def perception_step(Rover):
    # Perform perception steps to update Rover()
    # NOTE: camera image is coming to you in Rover.img

    # 1) Define source and destination points for perspective transform
    if not hasattr(Rover, 'source'):
        Rover.source = np.float32([[14, 140], [301 ,140],[200, 96], [118, 96]])
    if not hasattr(Rover, 'destination'):
        im_shape = Rover.img.shape
        dst_size_half = 5
        Rover.destination = np.float32([
            [im_shape[1]/2 - dst_size_half, im_shape[0]],
            [im_shape[1]/2 + dst_size_half, im_shape[0]],
            [im_shape[1]/2 + dst_size_half, im_shape[0] - 2*dst_size_half],
            [im_shape[1]/2 - dst_size_half, im_shape[0] - 2*dst_size_half]])

    # 2) Apply perspective transform
    warped = perspect_transform(Rover.img, Rover.source, Rover.destination)

    # 3) Apply color threshold to identify navigable terrain, obstacles, and rock samples
    navigable_terrain = color_thresh(warped)
    # obstacles = -(navigable_terrain - 1)
    obstacles = color_thresh(warped, rgb_thresh_min=(0, 0, 0), rgb_thresh_max=(120, 120, 120))
    rock_samples = color_thresh(warped, rgb_thresh_min=(100, 100, 0), rgb_thresh_max=(255,255,50))

    # 4) Update Rover.vision_image (this will be displayed on left side of screen)
    Rover.vision_image[:,:,2] = navigable_terrain * 255
    Rover.vision_image[:,:,0] = obstacles * 255
    Rover.vision_image[:,:,1] = rock_samples * 255

    # 5) Convert map image pixel values to rover-centric coords
    nav_x, nav_y = rover_coords(navigable_terrain)
    obs_x, obs_y = rover_coords(obstacles)
    roc_x, roc_y = rover_coords(rock_samples)

    # 6) Convert rover-centric pixel values to world coordinates
    world_scale = 10
    nav_x_world, nav_y_world = pix_to_world(nav_x, nav_y, Rover.pos[0], Rover.pos[1],
                                            Rover.yaw, Rover.worldmap.shape[0], world_scale)
    obs_x_world, obs_y_world = pix_to_world(obs_x, obs_y, Rover.pos[0], Rover.pos[1],
                                            Rover.yaw, Rover.worldmap.shape[0], world_scale)
    roc_y_world, roc_x_world = pix_to_world(roc_x, roc_y, Rover.pos[0], Rover.pos[1],
                                            Rover.yaw, Rover.worldmap.shape[0], world_scale)

    # 7) Update Rover worldmap (to be displayed on right side of screen)
    update_strength = 1
    Rover.worldmap[nav_y_world, nav_x_world, 2] += update_strength
    Rover.worldmap[obs_y_world, obs_x_world, 0] += update_strength
    Rover.worldmap[roc_y_world, roc_x_world, 1] += update_strength

    # 8) Convert rover-centric pixel positions to polar coordinates
    Rover.nav_dists, Rover.nav_angles = to_polar_coords(nav_x, nav_y)
    Rover.roc_dists, Rover.roc_angles = to_polar_coords(roc_x, roc_y)

    return Rover
