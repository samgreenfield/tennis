import cv2
import os
import pickle
from tqdm import tqdm
from ultralytics import YOLO
from .geometry_utils import euclidean_distance, bbox_feet, bbox_center

class PlayerTracker():
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, stub_path = "stubs/player_stub.pkl"):
        if os.path.isfile(stub_path):
            with open(stub_path, 'rb') as stub:
                return pickle.load(stub)
        print("Detecting players in frames...")
            
        player_detections = []
        for frame in tqdm(frames, desc="Detecting players", unit="frame"):
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        with open(stub_path, 'wb') as stub:
            pickle.dump(player_detections, stub)

        return player_detections
    
    def detect_frame(self, frame):
        results = self.model.track(frame, persist = True, verbose = False)[0]
        class_names = results.names
        player_dict = {}

        for box in results.boxes:
            if box.id is not None:
                track_id = int(box.id.tolist()[0])
                result = box.xyxy.tolist()[0]
                class_ids = box.cls.tolist()[0]
                det_class_names = class_names[class_ids]

                if det_class_names == "person":
                    player_dict[track_id] = result

        return player_dict

    def choose_and_filter_players(self, court_keypoints, player_detections, homography_obj, homography):
        player_detections_first_frame = player_detections[0]
        chosen_players, _ = self.choose_players(court_keypoints, player_detections_first_frame, homography_obj, homography)
        filtered_player_detections = []
        for player_detection in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_detection.items() if track_id in chosen_players}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict, homography_obj, homography):
        distances = []

        for track_id, bbox in player_dict.items():
            player_feet = bbox_feet(bbox)
            min_distance = float('inf')
            for idx, keypoint in enumerate(court_keypoints[0]):
                distance_to_keypoint = euclidean_distance(homography_obj.map_point(player_feet, homography), homography_obj.map_point(keypoint, homography))
                if distance_to_keypoint < min_distance:
                    min_distance = distance_to_keypoint
                    min_point = idx

            distances.append((track_id, min_distance, min_point))

        distances.sort(key=lambda x: x[1], reverse=False)
        chosen_players = [player_id for player_id, min_distance, min_point in distances[:2]]
        return chosen_players, distances
    
    def draw_bboxes(self, video_frames, player_detections, scale):
        output_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (73, 247, 245), int(3 * scale))
                # frame = cv2.putText(frame, str(track_id), bbox_feet(bbox), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            output_frames.append(frame)
        return output_frames

def map_player_detections(player_detections, homography_obj, homographies):
        mapped_player_detections = []
        for frame_idx, H in enumerate(homographies):
            if player_detections[frame_idx]:
                frame_dict = {}
                for track_id, player_detection in player_detections[frame_idx].items():
                    player_point = bbox_feet(player_detection)
                    player_mapped = homography_obj.map_point(player_point, H)
                    player_mapped = (int(player_mapped[0]), int(player_mapped[1]))
                    frame_dict[track_id] = player_mapped
                mapped_player_detections.append(frame_dict)
            else:
                mapped_player_detections.append({})
        return mapped_player_detections

def tether_players_to_points(player_tracker, frame_points, player_detections, homography, homographies, frames, interpolated_points_per_frame):
    _, distances = player_tracker.choose_players(frame_points, player_detections[0], homography, homographies[0])
    for frame_num, frame in enumerate(frames):
        for track_id, min_distance, min_point in distances:
            point = interpolated_points_per_frame[frame_num][min_point]
            if track_id in player_detections[frame_num].keys():
                frame = cv2.line(frame, bbox_feet(player_detections[frame_num][track_id]), (int(point[0]), int(point[1])), (255, 0, 0))