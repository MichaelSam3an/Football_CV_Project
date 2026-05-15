from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys
from pathlib import Path
import torch

sys.path.append("../")

from utils import get_center_of_bbox, get_bbox_width, get_foot_position


DEVICE = 0 if torch.cuda.is_available() else "cpu"


class Tracker:
    def __init__(
        self,
        model_path,
        ball_model_path="models/yolov8s_ball_best.pt",
        use_new_ball_model=True
    ):
        self.model_path = model_path
        self.ball_model_path = ball_model_path
        self.device = DEVICE

        # Main YOLO model: players + referees + goalkeeper
        self.model = YOLO(model_path)

        # Optional custom YOLO model: ball only
        self.use_new_ball_model = use_new_ball_model
        self.ball_model = None

        if self.use_new_ball_model:
            if not Path(ball_model_path).exists():
                raise FileNotFoundError(
                    f"Ball model not found: {ball_model_path}\n"
                    f"Put yolov8s_ball_best.pt inside the models folder."
                )
            self.ball_model = YOLO(ball_model_path)

        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.20,
            lost_track_buffer=120,
            minimum_matching_threshold=0.65,
            frame_rate=25
         )
        
        self.ball_switch_candidate = None
        self.ball_switch_frames = 0
        # Detection settings
        self.main_conf = 0.15
        self.main_imgsz = 960

        self.ball_conf = 0.10
        self.ball_imgsz = 960

        # Ball filtering settings
        self.max_ball_interpolation_gap = 8
        self.max_ball_box_width_ratio = 0.12
        self.max_ball_box_height_ratio = 0.12
        self.max_ball_jump_ratio = 0.35

        # Play-area margins
        self.play_area_left_ratio = 0.01
        self.play_area_right_ratio = 0.99
        self.play_area_top_ratio = 0.03
        self.play_area_bottom_ratio = 0.98

    def add_position_to_tracks(self, tracks):
        for object_name, object_tracks in tracks.items():
            if object_name not in ["players", "referees", "ball"]:
                continue

            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info["bbox"]

                    if object_name == "ball":
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)

                    tracks[object_name][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions, max_gap=None):
        if max_gap is None:
            max_gap = self.max_ball_interpolation_gap

        clean_ball_positions = []

        for frame_ball in ball_positions:
            bbox = frame_ball.get(1, {}).get("bbox", None)

            if bbox is None or len(bbox) != 4:
                clean_ball_positions.append([np.nan, np.nan, np.nan, np.nan])
            else:
                clean_ball_positions.append(bbox)

        df = pd.DataFrame(clean_ball_positions, columns=["x1", "y1", "x2", "y2"])

        df = df.interpolate(
            method="linear",
            limit=max_gap,
            limit_direction="both"
        )

        new_ball_positions = []

        for row in df.to_numpy().tolist():
            if any(pd.isna(value) for value in row):
                new_ball_positions.append({})
            else:
                new_ball_positions.append({1: {"bbox": row}})

        return new_ball_positions

    def detect_frames(self, frames):
        batch_size = 2
        detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size],
                conf=self.main_conf,
                device=self.device,
                imgsz=self.main_imgsz,
                verbose=False
            )
            detections += detections_batch

        return detections

    def detect_ball_frames(self, frames):
        if not self.use_new_ball_model or self.ball_model is None:
            return [None for _ in frames]

        batch_size = 2
        ball_detections = []

        for i in range(0, len(frames), batch_size):
            detections_batch = self.ball_model.predict(
                frames[i:i + batch_size],
                conf=self.ball_conf,
                device=self.device,
                imgsz=self.ball_imgsz,
                verbose=False
            )
            ball_detections += detections_batch

        return ball_detections

    def bbox_iou(self, box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)

        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    def is_reasonable_ball_size(self, bbox, frame_shape):
        frame_height, frame_width = frame_shape[:2]

        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1

        if bbox_width <= 1 or bbox_height <= 1:
            return False

        if bbox_width > frame_width * self.max_ball_box_width_ratio:
            return False

        if bbox_height > frame_height * self.max_ball_box_height_ratio:
            return False

        return True

    def is_inside_play_area(self, bbox, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x, y = get_center_of_bbox(bbox)

        left_margin = frame_width * self.play_area_left_ratio
        right_margin = frame_width * self.play_area_right_ratio
        top_margin = frame_height * self.play_area_top_ratio
        bottom_margin = frame_height * self.play_area_bottom_ratio

        if x < left_margin or x > right_margin:
            return False

        if y < top_margin or y > bottom_margin:
            return False

        return True

    def is_valid_ball_motion(self, current_bbox, previous_bbox, frame_shape):
        if previous_bbox is None:
            return True

        frame_height, frame_width = frame_shape[:2]
        frame_diag = (frame_width ** 2 + frame_height ** 2) ** 0.5

        current_center = get_center_of_bbox(current_bbox)
        previous_center = get_center_of_bbox(previous_bbox)

        dx = current_center[0] - previous_center[0]
        dy = current_center[1] - previous_center[1]
        distance = (dx ** 2 + dy ** 2) ** 0.5

        max_allowed_jump = frame_diag * self.max_ball_jump_ratio

        if distance <= max_allowed_jump:
            return True

        # Allow larger jumps for aerial balls
        current_height = current_bbox[3] - current_bbox[1]
        previous_height = previous_bbox[3] - previous_bbox[1]

        height_ratio = current_height / max(previous_height, 1)

        # Ball appears smaller in air
        if height_ratio < 0.7:
            return distance <= max_allowed_jump * 2.0

        return False

    def get_ball_bbox_from_ball_model(self, detection, frame):

        if detection is None or detection.boxes is None:
            return None

        best_bbox = None
        best_score = -999999

        previous_ball_bbox = getattr(
            self,
            "previous_ball_bbox",
            None
        )
    
        for box in detection.boxes:

            score = float(box.conf[0])
            bbox = box.xyxy[0].tolist()

            # =====================================
            # BASIC FILTERS
            # =====================================
    
            if not self.is_reasonable_ball_size(
                bbox,
                frame.shape
            ):
                continue

            if not self.is_inside_play_area(
                bbox,
                frame.shape
            ):
                continue

            combined_score = score

            # =====================================
            # DISTANCE CONSISTENCY
            # =====================================

            if previous_ball_bbox is not None:

                previous_center = get_center_of_bbox(
                    previous_ball_bbox
                )

                current_center = get_center_of_bbox(
                    bbox
                )

                distance = np.linalg.norm(
                    np.array(current_center) -
                    np.array(previous_center)
                )

                frame_diag = (
                    frame.shape[0] ** 2 +
                    frame.shape[1] ** 2
                ) ** 0.5

                normalized_distance = distance / frame_diag

                # Penalize huge jumps
                combined_score -= normalized_distance * 3.5

            # =====================================
            # SIZE CONSISTENCY
            # =====================================

            bbox_width = bbox[2] - bbox[0]
            bbox_height = bbox[3] - bbox[1]

            bbox_area = bbox_width * bbox_height

            # Reject absurd detections
            if bbox_area > 5000:
                aspect_ratio = bbox_width / max(bbox_height, 1)
                if aspect_ratio > 2.0 or aspect_ratio < 0.5:
                continue

            # Prefer realistic football sizes
            if 20 < bbox_area < 1200:
                combined_score += 0.2

            # =====================================
            # BEST DETECTION
            # =====================================

            if combined_score > best_score:
                best_score = combined_score
                best_bbox = bbox

        # =====================================
        # STORE PREVIOUS BALL
        # =====================================

        if best_bbox is not None:
            if self.previous_ball_bbox is not None:
                 previous_center = get_center_of_bbox(
                        self.previous_ball_bbox
                )
                 current_center = get_center_of_bbox(
                    best_bbox
                )
                distance = np.linalg.norm(
                    np.array(current_center) -
                    np.array(previous_center)
                )
                frame_diag = (
                    frame.shape[0] ** 2 +
                    frame.shape[1] ** 2
                ) ** 0.5
                normalized_distance = distance / frame_diag
                if normalized_distance > 0.20:
                    if self.ball_switch_candidate is None:
                        self.ball_switch_candidate = best_bbox
                        self.ball_switch_frames = 1
                    else:
                        self.ball_switch_frames += 1

                    if self.ball_switch_frames >= 3:
                        self.previous_ball_bbox = best_bbox
                        self.ball_switch_candidate = None
                        self.ball_switch_frames = 0

                 else:
                    self.previous_ball_bbox = best_bbox
                    self.ball_switch_candidate = None
                    self.ball_switch_frames = 0

         else:
            self.previous_ball_bbox = best_bbox

        return best_bbox

    def get_ball_bbox_from_normal_detection(self, detection, frame):
        if detection is None or detection.boxes is None:
            return None

        names = detection.names
        best_bbox = None
        best_score = 0

        for box in detection.boxes:
            cls_id = int(box.cls[0])
            score = float(box.conf[0])

            class_name = str(names.get(cls_id, "")).lower().strip()
            if class_name != "ball":
                continue

            bbox = box.xyxy[0].tolist()

            if not self.is_reasonable_ball_size(bbox, frame.shape):
                continue

            if not self.is_inside_play_area(bbox, frame.shape):
                continue

            if score > best_score:
                best_score = score
                best_bbox = bbox

        return best_bbox

    def get_previous_ball_bbox(self, tracks, frame_num, lookback=8):
        start = max(0, frame_num - lookback)

        for prev_frame_num in range(frame_num - 1, start - 1, -1):
            previous_bbox = tracks["ball"][prev_frame_num].get(1, {}).get("bbox", None)
            if previous_bbox is not None:
                return previous_bbox

        return None

    def remove_duplicate_player_referee_boxes(self, detection_supervision, cls_names_inv):
        if len(detection_supervision) == 0:
            return detection_supervision

        if "player" not in cls_names_inv or "referee" not in cls_names_inv:
            return detection_supervision

        player_id = cls_names_inv["player"]
        referee_id = cls_names_inv["referee"]

        boxes = detection_supervision.xyxy
        class_ids = detection_supervision.class_id

        keep = np.ones(len(detection_supervision), dtype=bool)

        referee_indices = np.where(class_ids == referee_id)[0]
        player_indices = np.where(class_ids == player_id)[0]

        for p_idx in player_indices:
            p_box = boxes[p_idx]
            for r_idx in referee_indices:
                r_box = boxes[r_idx]
                iou = self.bbox_iou(p_box, r_box)
                if iou > 0.35:
                    keep[p_idx] = False
                    break

        return detection_supervision[keep]

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        ball_detections = self.detect_ball_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        new_ball_model_used = 0
        normal_ball_used = 0
        missed_ball = 0
        rejected_by_motion = 0
        rejected_by_filter = 0

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            if "goalkeeper" in cls_names_inv and "player" in cls_names_inv:
                goalkeeper_id = cls_names_inv["goalkeeper"]
                player_id = cls_names_inv["player"]

                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if class_id == goalkeeper_id:
                        detection_supervision.class_id[object_ind] = player_id

            allowed_class_ids = []

            if "player" in cls_names_inv:
                allowed_class_ids.append(cls_names_inv["player"])

            if "referee" in cls_names_inv:
                allowed_class_ids.append(cls_names_inv["referee"])

            if len(allowed_class_ids) > 0 and len(detection_supervision) > 0:
                mask = np.isin(detection_supervision.class_id, allowed_class_ids)
                detection_supervision = detection_supervision[mask]

            detection_supervision = self.remove_duplicate_player_referee_boxes(
                detection_supervision=detection_supervision,
                cls_names_inv=cls_names_inv
            )

            detection_with_tracks = self.tracker.update_with_detections(
                detection_supervision
            )

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = int(frame_detection[4])

                if "player" in cls_names_inv and cls_id == cls_names_inv["player"]:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if "referee" in cls_names_inv and cls_id == cls_names_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            ball_bbox = None
            ball_source = None

            if self.use_new_ball_model:
                ball_bbox = self.get_ball_bbox_from_ball_model(
                    ball_detections[frame_num],
                    frames[frame_num]
                )
                if ball_bbox is not None:
                    ball_source = "custom"

            if ball_bbox is None:
                ball_bbox = self.get_ball_bbox_from_normal_detection(
                    detection,
                    frames[frame_num]
                )
                if ball_bbox is not None:
                    ball_source = "normal"

            if ball_bbox is not None:
                previous_ball_bbox = self.get_previous_ball_bbox(
                    tracks=tracks,
                    frame_num=frame_num,
                    lookback=8
                )

                valid_motion = self.is_valid_ball_motion(
                    current_bbox=ball_bbox,
                    previous_bbox=previous_ball_bbox,
                    frame_shape=frames[frame_num].shape
                )

                if valid_motion:
                    tracks["ball"][frame_num][1] = {"bbox": ball_bbox}

                    if ball_source == "custom":
                        new_ball_model_used += 1
                    elif ball_source == "normal":
                        normal_ball_used += 1
                else:
                    rejected_by_motion += 1
                    missed_ball += 1
            else:
                missed_ball += 1

        print(
            f"Ball detection | custom model: {new_ball_model_used}, "
            f"normal fallback: {normal_ball_used}, "
            f"missed: {missed_ball}, "
            f"rejected motion: {rejected_by_motion}, "
            f"rejected filter: {rejected_by_filter}"
        )

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    
    def get_ui_scale(self, frame):
        h, w = frame.shape[:2]
        return w / 1920.0    
    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        # Scale based on frame resolution
        scale = self.get_ui_scale(frame)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=max(1, int(2 * scale)),
            lineType=cv2.LINE_AA
        )

        rect_w = int(28 * scale)
        rect_h = int(16 * scale)

        x1 = int(x_center - rect_w // 2)
        x2 = int(x_center + rect_w // 2)

        y1 = int(y2 + 8 * scale)
        y2_rect = int(y1 + rect_h)

        if track_id is not None:
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2_rect),
                color,
                cv2.FILLED
            )

            font_scale = 0.55 * scale
            thickness = max(1, int(2 * scale))

            text = str(track_id)

            (tw, th), _ = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                thickness
            )

            text_x = int(x_center - tw // 2)
            text_y = int(y1 + th + 2 * scale)

            cv2.putText(
                frame,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])

        x, _ = get_center_of_bbox(bbox)

        scale = self.get_ui_scale(frame)

        triangle_points = np.array([
            [x, y],
            [x - int(10 * scale), y - int(20 * scale)],
            [x + int(10 * scale), y - int(20 * scale)],
        ])

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            color,
            cv2.FILLED
        )

        cv2.drawContours(
            frame,
            [triangle_points],
            0,
            (0, 0, 0),
            max(1, int(2 * scale))
        )

        return frame

    def draw_traingle(self, frame, bbox, color):
        return self.draw_triangle(frame, bbox, color)

    def draw_ball_marker(self, frame, bbox, color=(0, 255, 0)):
        cx, cy = get_center_of_bbox(bbox)

        scale = self.get_ui_scale(frame)

        size = max(4, int(8 * scale))

        cv2.circle(
            frame,
            (int(cx), int(cy)),
            size,
            color,
            2,
            cv2.LINE_AA
        )

        cv2.circle(
            frame,
            (int(cx), int(cy)),
            2,
            color,
            -1
        )

        return frame

    def draw_team_ball_control(self, frame, global_frame_num, team_ball_control):
        overlay = frame.copy()

        height, width = frame.shape[:2]

        x1 = int(width * 0.74)
        y1 = int(height * 0.82)

        x2 = int(width * 0.98)
        y2 = int(height * 0.95)

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            -1
        )

        cv2.addWeighted(
            overlay,
            0.35,
            frame,
            0.65,
            0,
            frame
        )

        if len(team_ball_control) == 0:
            team_1 = 0
            team_2 = 0
        else:
            team_ball_control_till_frame = team_ball_control[:global_frame_num + 1]

            team_1_num_frames = team_ball_control_till_frame[
                team_ball_control_till_frame == 1
            ].shape[0]

            team_2_num_frames = team_ball_control_till_frame[
                team_ball_control_till_frame == 2
            ].shape[0]

            total_frames = team_1_num_frames + team_2_num_frames

            if total_frames == 0:
                team_1 = 0
                team_2 = 0
            else:
                team_1 = team_1_num_frames / total_frames
                team_2 = team_2_num_frames / total_frames

        font_scale = 0.55 * self.get_ui_scale(frame)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {team_1 * 100:.2f}%",
            (x1 + 12, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Team 2 Ball Control: {team_2 * 100:.2f}%",
            (x1 + 12, y1 + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

        return frame

    def draw_annotations(
        self,
        video_frames,
        tracks,
        team_ball_control,
        frame_offset=0,
        game_state_per_frame=None
    ):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referees"][frame_num]

            for track_id, player in player_dict.items():
                color = player.get("team_color", (0, 0, 255))

                frame = self.draw_ellipse(
                    frame,
                    player["bbox"],
                    color,
                    track_id
                )

                if player.get("has_ball", False):
                    frame = self.draw_triangle(
                        frame,
                        player["bbox"],
                        (0, 0, 255)
                    )

            for _, referee in referee_dict.items():
                frame = self.draw_ellipse(
                    frame,
                    referee["bbox"],
                    (0, 255, 255)
                )

            for _, ball in ball_dict.items():
                frame = self.draw_ball_marker(
                    frame,
                    ball["bbox"],
                    (0, 255, 0)
                )

            global_frame_num = frame_offset + frame_num

            frame = self.draw_team_ball_control(
                frame,
                global_frame_num,
                team_ball_control
            )

            output_video_frames.append(frame)

        return output_video_frames
