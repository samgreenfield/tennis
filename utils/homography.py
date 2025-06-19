import numpy as np
import cv2
import os

class Homography():
    def __init__(self):
        self.court_width = 319
        self.court_height = 670
        self.offset_x = 87
        self.offset_y = 35

        self.court_image_path = 'media/tennis_court.png'
        if os.path.isfile('media/tennis_court.png'):
            self.court_image = cv2.imread('media/tennis_court.png')
        else: print("No image found")
        
        self.court_pts = np.array([
            [self.offset_x, self.offset_y],
            [self.offset_x + self.court_width, self.offset_y],
            [self.offset_x + self.court_width, self.offset_y + self.court_height],  
            [self.offset_x, self.offset_y + self.court_height]
        ], dtype=np.float32)

    def compute_homography(self, corner_points):
        ordered_corners = np.array([
            corner_points[0],  # top-left
            corner_points[1],  # top-right
            corner_points[3],  # bottom-right
            corner_points[2],  # bottom-left
        ], dtype=np.float32)

        H, status = cv2.findHomography(ordered_corners, self.court_pts)
        return H
    
    def map_point(self, point, H):
        pt = np.array([ [point[0], point[1]] ], dtype=np.float32)
        pt = np.array([pt])
        mapped_pt = cv2.perspectiveTransform(pt, H)
        return mapped_pt[0][0]