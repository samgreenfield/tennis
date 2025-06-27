import numpy as np
import cv2
import os

class Homography():
    def __init__(self):
        self.court_width = 239
        self.court_height = 504
        self.offset_x = 130
        self.offset_y = 123

        self.court_image_path = 'media/tennis_court.png'
        if os.path.isfile('media/tennis_court.png'):
            self.court_image = cv2.imread('media/tennis_court.png')
        else: print("No image found")
        
        self.ref_court_pts = np.array([
        [130, 123],
        [369, 123],
        [130, 627],
        [369, 627],
        [155.5, 123],
        [155.5, 627],
        [344, 123],
        [344, 627],
        [155.5, 247.5],
        [344, 247.5],
        [155.5, 502.5],
        [344, 502.5],
        [250, 247.5],
        [250, 502.5]], dtype=np.float32)

    def compute_homography(self, inferred_points):
        inferred_points = np.array(inferred_points, dtype=np.float32)

        H, status = cv2.findHomography(inferred_points, self.ref_court_pts)
        return H
    
    def map_point(self, point, H):
        pt = np.array([ [point[0], point[1]] ], dtype=np.float32)
        pt = np.array([pt])
        mapped_pt = cv2.perspectiveTransform(pt, H)
        return mapped_pt[0][0]