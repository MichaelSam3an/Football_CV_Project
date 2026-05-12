import cv2
import pickle
import os
import numpy as np


class CameraMovementEstimator:
    def __init__(self, frame):
        self.minimum_distance = 5
        self.enable_hud = True

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,
                0.03
            )
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(
        self,
        frames,
        read_from_stub=False,
        stub_path=None
    ):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0, 0] for _ in range(len(frames))]

        if len(frames) == 0:
            return camera_movement

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        old_features = cv2.goodFeaturesToTrack(
            old_gray,
            **self.features
        )

        if old_features is None or len(old_features) == 0:
            print("Warning: no camera features found in first frame. Camera movement disabled for this chunk.")

            if stub_path is not None:
                with open(stub_path, "wb") as f:
                    pickle.dump(camera_movement, f)

            return camera_movement

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(
                    old_gray,
                    **self.features
                )

                if old_features is None or len(old_features) == 0:
                    camera_movement[frame_num] = camera_movement[frame_num - 1]
                    old_gray = frame_gray.copy()
                    continue

            try:
                new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray,
                    frame_gray,
                    old_features,
                    None,
                    **self.lk_params
                )
            except cv2.error:
                camera_movement[frame_num] = camera_movement[frame_num - 1]
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(
                    old_gray,
                    **self.features
                )
                continue

            if new_features is None or status is None:
                camera_movement[frame_num] = camera_movement[frame_num - 1]
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(
                    old_gray,
                    **self.features
                )
                continue

            status = status.reshape(-1)

            valid_old_features = old_features[status == 1]
            valid_new_features = new_features[status == 1]

            if len(valid_old_features) == 0 or len(valid_new_features) == 0:
                camera_movement[frame_num] = camera_movement[frame_num - 1]
                old_gray = frame_gray.copy()
                old_features = cv2.goodFeaturesToTrack(
                    old_gray,
                    **self.features
                )
                continue

            max_distance = 0
            camera_movement_x = 0
            camera_movement_y = 0

            for old, new in zip(valid_old_features, valid_new_features):
                old_point = old.ravel()
                new_point = new.ravel()

                distance = np.linalg.norm(new_point - old_point)

                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x = new_point[0] - old_point[0]
                    camera_movement_y = new_point[1] - old_point[1]

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [
                    camera_movement_x,
                    camera_movement_y
                ]

                old_features = cv2.goodFeaturesToTrack(
                    frame_gray,
                    **self.features
                )
            else:
                camera_movement[frame_num] = camera_movement[frame_num - 1]

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(
        self,
        tracks,
        camera_movement_per_frame
    ):
        for object_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                if frame_num >= len(camera_movement_per_frame):
                    continue

                for track_id, track_info in track.items():
                    position = track_info.get("position", None)

                    if position is None:
                        continue

                    camera_movement = camera_movement_per_frame[frame_num]

                    position_adjusted = (
                        position[0] - camera_movement[0],
                        position[1] - camera_movement[1]
                    )

                    tracks[object_name][frame_num][track_id]["position_adjusted"] = position_adjusted

    def draw_camera_movement(
        self,
        frames,
        camera_movement_per_frame
    ):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            if not self.enable_hud:
                output_frames.append(frame)
                continue

            if frame_num < len(camera_movement_per_frame):
                x_movement, y_movement = camera_movement_per_frame[frame_num]
            else:
                x_movement, y_movement = 0, 0

            overlay = frame.copy()
            x1, y1 = 20, 20
            x2, y2 = 340, 90

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 0, 0),
                -1
            )

            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

            cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (30, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (30, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            output_frames.append(frame)

        return output_frames
