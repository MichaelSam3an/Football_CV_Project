from trackers import Tracker
import cv2
import numpy as np
import gc
import os
import csv
import argparse
from pathlib import Path
from tqdm import tqdm

from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer

# Speed disabled for now
# from speed_and_distance_estimator import SpeedAndDistance_Estimator

from analytics import MatchAnalytics


class PlayerIDStabilizer:
    """
    Stabilizes player/referee IDs across chunks.

    This does not replace ByteTrack. It sits after ByteTrack and remaps local
    ByteTrack IDs into more stable global IDs.
    """

    def __init__(
        self,
        max_missing_frames=120,
        max_distance_ratio=0.08,
        max_hard_distance_ratio=0.20,
        min_iou=0.01
    ):
        self.max_missing_frames = max_missing_frames
        self.max_distance_ratio = max_distance_ratio
        self.max_hard_distance_ratio = max_hard_distance_ratio
        self.min_iou = min_iou

        self.next_global_id = {
            "players": 1,
            "referees": 1
        }

        self.local_to_global = {
            "players": {},
            "referees": {}
        }

        self.global_memory = {
            "players": {},
            "referees": {}
        }

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2
        ], dtype=np.float32)

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

    def get_distance(self, bbox_a, bbox_b):
        center_a = self.get_bbox_center(bbox_a)
        center_b = self.get_bbox_center(bbox_b)
        return float(np.linalg.norm(center_a - center_b))

    def create_new_global_id(self, object_name):
        global_id = self.next_global_id[object_name]
        self.next_global_id[object_name] += 1
        return global_id

    def find_best_global_match(
        self,
        object_name,
        local_track,
        global_frame_num,
        frame_width,
        frame_height,
        used_global_ids
    ):
        current_bbox = local_track["bbox"]
        current_team = local_track.get("team", None)

        frame_diag = (frame_width ** 2 + frame_height ** 2) ** 0.5
        max_distance = frame_diag * self.max_distance_ratio
        max_hard_distance = frame_diag * self.max_hard_distance_ratio

        best_global_id = None
        best_score = -999999

        for global_id, memory in self.global_memory[object_name].items():
            if global_id in used_global_ids:
                continue

            age = global_frame_num - memory["last_seen"]

            if age < 0 or age > self.max_missing_frames:
                continue

            previous_bbox = memory["bbox"]
            previous_team = memory.get("team", None)

            if (
                object_name == "players"
                and current_team is not None
                and previous_team is not None
                and current_team != previous_team
            ):
                continue

            distance = self.get_distance(current_bbox, previous_bbox)
            iou = self.bbox_iou(current_bbox, previous_bbox)

            if distance > max_hard_distance and iou < self.min_iou:
                continue

            distance_score = max(0.0, 1.0 - (distance / max_distance))
            iou_score = iou
            age_score = max(0.0, 1.0 - (age / self.max_missing_frames))

            team_score = 0.0
            if object_name == "players" and current_team is not None and previous_team is not None:
                if current_team == previous_team:
                    team_score = 0.5

            score = (
                distance_score * 2.0
                + iou_score * 3.0
                + age_score * 0.5
                + team_score
            )

            if score > best_score:
                best_score = score
                best_global_id = global_id

        return best_global_id

    def stabilize_object_tracks(
        self,
        tracks,
        object_name,
        frame_offset,
        frame_width,
        frame_height
    ):
        object_tracks = tracks[object_name]

        for frame_num, frame_tracks in enumerate(object_tracks):
            global_frame_num = frame_offset + frame_num

            stable_frame_tracks = {}
            used_global_ids = set()

            sorted_items = sorted(
                frame_tracks.items(),
                key=lambda item: item[0]
            )

            for local_id, track_info in sorted_items:
                local_id = int(local_id)

                global_id = None

                if local_id in self.local_to_global[object_name]:
                    candidate_global_id = self.local_to_global[object_name][local_id]

                    if candidate_global_id in self.global_memory[object_name]:
                        memory = self.global_memory[object_name][candidate_global_id]

                        age = global_frame_num - memory["last_seen"]

                        distance = self.get_distance(
                            track_info["bbox"],
                            memory["bbox"]
                        )

                        frame_diag = (frame_width ** 2 + frame_height ** 2) ** 0.5
                        hard_distance = frame_diag * self.max_hard_distance_ratio

                        if (
                            age <= self.max_missing_frames
                            and distance <= hard_distance
                            and candidate_global_id not in used_global_ids
                        ):
                            global_id = candidate_global_id

                if global_id is None:
                    global_id = self.find_best_global_match(
                        object_name=object_name,
                        local_track=track_info,
                        global_frame_num=global_frame_num,
                        frame_width=frame_width,
                        frame_height=frame_height,
                        used_global_ids=used_global_ids
                    )

                if global_id is None:
                    global_id = self.create_new_global_id(object_name)

                self.local_to_global[object_name][local_id] = global_id
                used_global_ids.add(global_id)

                track_info["original_track_id"] = local_id
                track_info["stable_track_id"] = global_id

                stable_frame_tracks[global_id] = track_info

                self.global_memory[object_name][global_id] = {
                    "bbox": track_info["bbox"],
                    "team": track_info.get("team", None),
                    "last_seen": global_frame_num
                }

            tracks[object_name][frame_num] = stable_frame_tracks

    def stabilize_tracks(
        self,
        tracks,
        frame_offset,
        frame_width,
        frame_height
    ):
        self.stabilize_object_tracks(
            tracks=tracks,
            object_name="players",
            frame_offset=frame_offset,
            frame_width=frame_width,
            frame_height=frame_height
        )

        self.stabilize_object_tracks(
            tracks=tracks,
            object_name="referees",
            frame_offset=frame_offset,
            frame_width=frame_width,
            frame_height=frame_height
        )

        return tracks


class GameStateAnalyzer:
    """
    Decides whether a frame should be trusted for analytics.

    Important:
    - Detection still runs.
    - Ball drawing still runs.
    - Analytics possession/pass updates freeze when state is not LIVE_PLAY.
    """

    def __init__(
        self,
        fps=30,
        static_speed_ratio=0.003,
        static_player_ratio=0.70,
        stopped_static_frames=45,
        missing_ball_frames=60,
        outside_ball_frames=30,
        uncertain_after_missing_frames=12,
        max_ball_speed_ratio=0.090
    ):
        self.fps = fps

        self.static_speed_ratio = static_speed_ratio
        self.static_player_ratio = static_player_ratio

        self.stopped_static_frames = stopped_static_frames
        self.missing_ball_frames = missing_ball_frames
        self.outside_ball_frames = outside_ball_frames
        self.uncertain_after_missing_frames = uncertain_after_missing_frames

        self.max_ball_speed_ratio = max_ball_speed_ratio

        self.previous_player_positions = {}
        self.previous_ball_center = None

        self.static_frames = 0
        self.missing_ball_counter = 0
        self.outside_ball_counter = 0
        self.fast_ball_counter = 0

        self.last_state = "LIVE_PLAY"

    def get_bbox_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2
        ], dtype=np.float32)

    def is_ball_inside_safe_area(self, ball_bbox, frame_width, frame_height):
        if ball_bbox is None:
            return False

        cx, cy = self.get_bbox_center(ball_bbox)

        left = frame_width * 0.01
        right = frame_width * 0.99
        top = frame_height * 0.03
        bottom = frame_height * 0.98

        if cx < left or cx > right:
            return False

        if cy < top or cy > bottom:
            return False

        return True

    def update_player_static_state(self, player_track, frame_width, frame_height):
        if len(player_track) == 0:
            self.static_frames += 1
            return True

        frame_diag = (frame_width ** 2 + frame_height ** 2) ** 0.5
        static_threshold = frame_diag * self.static_speed_ratio

        static_count = 0
        comparable_count = 0

        current_positions = {}

        for player_id, track in player_track.items():
            bbox = track["bbox"]
            current_center = self.get_bbox_center(bbox)
            current_positions[player_id] = current_center

            if player_id in self.previous_player_positions:
                previous_center = self.previous_player_positions[player_id]
                distance = float(np.linalg.norm(current_center - previous_center))

                comparable_count += 1

                if distance < static_threshold:
                    static_count += 1

        self.previous_player_positions = current_positions

        if comparable_count == 0:
            return False

        static_ratio = static_count / comparable_count

        players_static = static_ratio >= self.static_player_ratio

        if players_static:
            self.static_frames += 1
        else:
            self.static_frames = 0

        return players_static

    def update_ball_state(self, ball_bbox, frame_width, frame_height):
        frame_diag = (frame_width ** 2 + frame_height ** 2) ** 0.5

        ball_detected = ball_bbox is not None

        if not ball_detected:
            self.missing_ball_counter += 1
            self.previous_ball_center = None
            return {
                "ball_detected": False,
                "ball_inside": False,
                "ball_fast": False
            }

        self.missing_ball_counter = 0

        ball_inside = self.is_ball_inside_safe_area(
            ball_bbox,
            frame_width,
            frame_height
        )

        if ball_inside:
            self.outside_ball_counter = 0
        else:
            self.outside_ball_counter += 1

        current_ball_center = self.get_bbox_center(ball_bbox)

        ball_fast = False

        if self.previous_ball_center is not None:
            distance = float(np.linalg.norm(current_ball_center - self.previous_ball_center))
            speed_ratio = distance / frame_diag

            if speed_ratio > self.max_ball_speed_ratio:
                ball_fast = True
                self.fast_ball_counter += 1
            else:
                self.fast_ball_counter = 0

        self.previous_ball_center = current_ball_center

        return {
            "ball_detected": True,
            "ball_inside": ball_inside,
            "ball_fast": ball_fast
        }

    def update(
        self,
        player_track,
        ball_bbox,
        frame_width,
        frame_height
    ):
        players_static = self.update_player_static_state(
            player_track=player_track,
            frame_width=frame_width,
            frame_height=frame_height
        )

        ball_info = self.update_ball_state(
            ball_bbox=ball_bbox,
            frame_width=frame_width,
            frame_height=frame_height
        )

        ball_detected = ball_info["ball_detected"]
        ball_inside = ball_info["ball_inside"]
        ball_fast = ball_info["ball_fast"]

        state = "LIVE_PLAY"
        reason = "normal"

        if self.missing_ball_counter >= self.missing_ball_frames:
            state = "STOPPED_PLAY"
            reason = "ball_missing_long"

        elif self.outside_ball_counter >= self.outside_ball_frames:
            state = "STOPPED_PLAY"
            reason = "ball_outside_long"

        elif self.static_frames >= self.stopped_static_frames and not ball_detected:
            state = "STOPPED_PLAY"
            reason = "players_static_ball_missing"

        elif self.static_frames >= self.stopped_static_frames and not ball_inside:
            state = "STOPPED_PLAY"
            reason = "players_static_ball_outside"

        elif self.missing_ball_counter >= self.uncertain_after_missing_frames:
            state = "UNCERTAIN"
            reason = "ball_missing_short"

        elif ball_detected and not ball_inside:
            state = "UNCERTAIN"
            reason = "ball_outside_short"

        elif ball_fast:
            state = "UNCERTAIN"
            reason = "fast_or_air_ball"

        else:
            state = "LIVE_PLAY"
            reason = "normal"

        self.last_state = state

        return {
            "state": state,
            "reason": reason,
            "players_static": players_static,
            "ball_detected": ball_detected,
            "ball_inside": ball_inside,
            "ball_fast": ball_fast,
            "missing_ball_counter": self.missing_ball_counter,
            "outside_ball_counter": self.outside_ball_counter,
            "static_frames": self.static_frames
        }


def protect_referees_from_team_logic(tracks):
    """
    Safety cleanup.

    Referees should not have team, team_color, or has_ball.
    """

    for frame_num, referee_track in enumerate(tracks["referees"]):
        for referee_id, referee in referee_track.items():
            referee.pop("team", None)
            referee.pop("team_color", None)
            referee.pop("has_ball", None)

    return tracks


def safe_get_camera_movement(video_frames):
    """
    Prevents OpenCV optical-flow errors from crashing a long video.

    If camera movement estimation fails, returns zero movement for the chunk.
    """

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    try:
        camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
            video_frames,
            read_from_stub=False,
            stub_path=None
        )
    except cv2.error as error:
        print("Camera movement estimation failed in this chunk.")
        print(f"OpenCV error: {error}")
        camera_movement_per_frame = [[0, 0] for _ in video_frames]
    except Exception as error:
        print("Camera movement estimation failed in this chunk.")
        print(f"Error: {error}")
        camera_movement_per_frame = [[0, 0] for _ in video_frames]

    return camera_movement_estimator, camera_movement_per_frame


def assign_team_colors_to_players(
    tracks,
    video_frames,
    team_assigner,
    team_colors_ready
):
    if not team_colors_ready:
        for frame_num, player_track in enumerate(tracks["players"]):
            if len(player_track) > 0:
                team_assigner.assign_team_color(
                    video_frames[frame_num],
                    player_track
                )

                team_colors_ready = True
                break

    if team_colors_ready:
        for frame_num, player_track in enumerate(tracks["players"]):
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num],
                    track["bbox"],
                    player_id
                )

                tracks["players"][frame_num][player_id]["team"] = team
                tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

    return tracks, team_colors_ready


def update_possession_with_game_state(
    tracks,
    player_assigner,
    ball_control_history,
    last_team_control,
    possession_state,
    game_state_per_frame,
    min_confirm_frames=4
):
    """
    Keeps your existing ball-to-player assigner.

    The only difference:
    - if game state is not LIVE_PLAY, we freeze analytics.
    - no lower-body rule is added here.
    """

    chunk_team_control = []

    for frame_num, player_track in enumerate(tracks["players"]):
        state_info = game_state_per_frame[frame_num]
        state = state_info["state"]

        if state != "LIVE_PLAY":
            chunk_team_control.append(last_team_control)
            continue

        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", None)

        if ball_bbox is None or len(ball_bbox) == 0:
            chunk_team_control.append(last_team_control)
            continue

        assigned_player = player_assigner.assign_ball_to_player(
            player_track,
            ball_bbox
        )

        assigned_team = 0

        if assigned_player != -1 and assigned_player in tracks["players"][frame_num]:
            assigned_team = tracks["players"][frame_num][assigned_player].get("team", 0)

        if assigned_player == -1 or assigned_team == 0:
            chunk_team_control.append(last_team_control)
            continue

        if assigned_player == possession_state["candidate_player"]:
            possession_state["candidate_frames"] += 1
        else:
            possession_state["candidate_player"] = assigned_player
            possession_state["candidate_team"] = assigned_team
            possession_state["candidate_frames"] = 1

        if possession_state["candidate_frames"] >= min_confirm_frames:
            possession_state["confirmed_player"] = possession_state["candidate_player"]
            possession_state["confirmed_team"] = possession_state["candidate_team"]

        confirmed_player = possession_state["confirmed_player"]
        confirmed_team = possession_state["confirmed_team"]

        if confirmed_player != -1 and confirmed_player in tracks["players"][frame_num]:
            tracks["players"][frame_num][confirmed_player]["has_ball"] = True

        if confirmed_team != 0:
            last_team_control = confirmed_team

        chunk_team_control.append(last_team_control)

    ball_control_history.extend(chunk_team_control)

    return last_team_control


def save_game_state_csv(game_state_history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "game_state_debug.csv")

    fieldnames = [
        "frame",
        "state",
        "reason",
        "players_static",
        "ball_detected",
        "ball_inside",
        "ball_fast",
        "missing_ball_counter",
        "outside_ball_counter",
        "static_frames"
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in game_state_history:
            writer.writerow(row)

    print(f"Game-state debug CSV saved to: {output_path}")


def process_chunk(
    video_frames,
    tracker,
    team_assigner,
    player_assigner,
    ball_control_history,
    last_team_control,
    team_colors_ready,
    frame_offset,
    analytics,
    frame_width,
    frame_height,
    id_stabilizer,
    possession_state,
    game_state_analyzer,
    game_state_history
):
    if len(video_frames) == 0:
        return [], last_team_control, team_colors_ready

    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,
        stub_path=None
    )

    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator, camera_movement_per_frame = safe_get_camera_movement(
        video_frames
    )

    try:
        camera_movement_estimator.add_adjust_positions_to_tracks(
            tracks,
            camera_movement_per_frame
        )
    except Exception as error:
        print("Could not add adjusted camera positions for this chunk.")
        print(f"Error: {error}")

    view_transformer = ViewTransformer()

    try:
        view_transformer.add_transformed_position_to_tracks(tracks)
    except Exception as error:
        print("View transformation failed for this chunk.")
        print(f"Error: {error}")

    # Interpolate short gaps only.
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    tracks, team_colors_ready = assign_team_colors_to_players(
        tracks=tracks,
        video_frames=video_frames,
        team_assigner=team_assigner,
        team_colors_ready=team_colors_ready
    )

    # Safety: referees must never be treated like team players.
    tracks = protect_referees_from_team_logic(tracks)

    # Stabilize after team assignment because team helps prevent wrong ID merges.
    tracks = id_stabilizer.stabilize_tracks(
        tracks=tracks,
        frame_offset=frame_offset,
        frame_width=frame_width,
        frame_height=frame_height
    )

    # Safety again after ID stabilization.
    tracks = protect_referees_from_team_logic(tracks)

    game_state_per_frame = []

    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num].get(1, {}).get("bbox", None)

        state_info = game_state_analyzer.update(
            player_track=player_track,
            ball_bbox=ball_bbox,
            frame_width=frame_width,
            frame_height=frame_height
        )

        global_frame_num = frame_offset + frame_num

        state_info_for_csv = {
            "frame": global_frame_num,
            "state": state_info["state"],
            "reason": state_info["reason"],
            "players_static": state_info["players_static"],
            "ball_detected": state_info["ball_detected"],
            "ball_inside": state_info["ball_inside"],
            "ball_fast": state_info["ball_fast"],
            "missing_ball_counter": state_info["missing_ball_counter"],
            "outside_ball_counter": state_info["outside_ball_counter"],
            "static_frames": state_info["static_frames"]
        }

        game_state_history.append(state_info_for_csv)
        game_state_per_frame.append(state_info)

    last_team_control = update_possession_with_game_state(
        tracks=tracks,
        player_assigner=player_assigner,
        ball_control_history=ball_control_history,
        last_team_control=last_team_control,
        possession_state=possession_state,
        game_state_per_frame=game_state_per_frame,
        min_confirm_frames=4
    )

    analytics.collect_chunk_data(
        tracks=tracks,
        frame_offset=frame_offset,
        frame_width=frame_width,
        frame_height=frame_height
    )

    global_team_ball_control = np.array(ball_control_history)

    output_video_frames = tracker.draw_annotations(
        video_frames,
        tracks,
        global_team_ball_control,
        frame_offset=frame_offset,
        game_state_per_frame=game_state_per_frame
    )

    try:
        output_video_frames = camera_movement_estimator.draw_camera_movement(
            output_video_frames,
            camera_movement_per_frame
        )
    except Exception as error:
        print("Could not draw camera movement for this chunk.")
        print(f"Error: {error}")

    return output_video_frames, last_team_control, team_colors_ready


def main():
    parser = argparse.ArgumentParser(description="Football CV pipeline")
    parser.add_argument(
        "--input_video",
        type=str,
        default="input_videos/match2.mp4",
        help="Path to the input video"
    )
    parser.add_argument(
        "--output_video",
        type=str,
        default="output_videos/2.mp4",
        help="Path to the output video"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=75,
        help="Number of frames per chunk"
    )

    args = parser.parse_args()

    INPUT_VIDEO_PATH = args.input_video
    OUTPUT_VIDEO_PATH = args.output_video
    CHUNK_SIZE = args.chunk_size
    OUTPUT_DIR = str(Path(OUTPUT_VIDEO_PATH).parent)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)

    if not cap.isOpened():
        print("Could not open video.")
        print(f"Check this path: {INPUT_VIDEO_PATH}")
        return

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        OUTPUT_VIDEO_PATH,
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        print("Could not create output video.")
        print(f"Check this path: {OUTPUT_VIDEO_PATH}")
        cap.release()
        return

    tracker = Tracker(
        model_path="models/yolov8s_players_refs_best.pt",
        ball_model_path="models/yolov8s_ball_best.pt",
        use_new_ball_model=True
    )

    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()

    analytics = MatchAnalytics(output_dir=OUTPUT_DIR)

    id_stabilizer = PlayerIDStabilizer(
        max_missing_frames=120,
        max_distance_ratio=0.08,
        max_hard_distance_ratio=0.20,
        min_iou=0.01
    )

    possession_state = {
        "confirmed_player": -1,
        "confirmed_team": 0,
        "candidate_player": -1,
        "candidate_team": 0,
        "candidate_frames": 0
    }

    game_state_analyzer = GameStateAnalyzer(
        fps=fps,
        static_speed_ratio=0.003,
        static_player_ratio=0.70,
        stopped_static_frames=int(fps * 1.5),
        missing_ball_frames=int(fps * 2.0),
        outside_ball_frames=int(fps * 1.0),
        uncertain_after_missing_frames=int(fps * 0.4),
        max_ball_speed_ratio=0.090
    )

    game_state_history = []

    team_colors_ready = False
    last_team_control = 0
    ball_control_history = []

    chunk_frames = []
    chunk_number = 1
    frame_offset = 0

    progress_bar = tqdm(
        total=total_video_frames,
        desc="Processing video",
        unit="frame"
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        chunk_frames.append(frame)
        progress_bar.update(1)

        if len(chunk_frames) == CHUNK_SIZE:
            print(
                f"\nProcessing chunk {chunk_number} | "
                f"frames {frame_offset + 1} to {frame_offset + len(chunk_frames)}"
            )

            output_frames, last_team_control, team_colors_ready = process_chunk(
                video_frames=chunk_frames,
                tracker=tracker,
                team_assigner=team_assigner,
                player_assigner=player_assigner,
                ball_control_history=ball_control_history,
                last_team_control=last_team_control,
                team_colors_ready=team_colors_ready,
                frame_offset=frame_offset,
                analytics=analytics,
                frame_width=width,
                frame_height=height,
                id_stabilizer=id_stabilizer,
                possession_state=possession_state,
                game_state_analyzer=game_state_analyzer,
                game_state_history=game_state_history
            )

            for output_frame in output_frames:
                out.write(output_frame)

            frame_offset += len(chunk_frames)

            chunk_frames.clear()
            del output_frames
            gc.collect()

            chunk_number += 1

    if len(chunk_frames) > 0:
        print(
            f"\nProcessing final chunk {chunk_number} | "
            f"frames {frame_offset + 1} to {frame_offset + len(chunk_frames)}"
        )

        output_frames, last_team_control, team_colors_ready = process_chunk(
            video_frames=chunk_frames,
            tracker=tracker,
            team_assigner=team_assigner,
            player_assigner=player_assigner,
            ball_control_history=ball_control_history,
            last_team_control=last_team_control,
            team_colors_ready=team_colors_ready,
            frame_offset=frame_offset,
            analytics=analytics,
            frame_width=width,
            frame_height=height,
            id_stabilizer=id_stabilizer,
            possession_state=possession_state,
            game_state_analyzer=game_state_analyzer,
            game_state_history=game_state_history
        )

        for output_frame in output_frames:
            out.write(output_frame)

        frame_offset += len(chunk_frames)

        chunk_frames.clear()
        del output_frames
        gc.collect()

    progress_bar.close()

    cap.release()
    out.release()

    analytics.save_all(ball_control_history)

    save_game_state_csv(
        game_state_history=game_state_history,
        output_dir=OUTPUT_DIR
    )

    print(f"Done. Output video saved to: {OUTPUT_VIDEO_PATH}")
    print(f"CSV, heatmap, and debug files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
