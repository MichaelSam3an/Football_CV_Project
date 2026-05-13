import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MatchAnalytics:
    def __init__(self, output_dir="output_videos"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.player_rows = []
        self.pass_events = []
        self.ball_rows = []

        self.last_player_with_ball = None

        # Heatmap tuning
        self.heatmap_bins = 100
        self.heatmap_blur_sigma = 2.2
        self.heatmap_min_relative = 0.0
        self.heatmap_max_relative = 1.0
        self.heatmap_use_live_play_only = True

    @staticmethod
    def _get_xy_from_position(position):
        if position is None:
            return None, None

        if not isinstance(position, (tuple, list, np.ndarray)):
            return None, None

        if len(position) < 2:
            return None, None

        x, y = position[0], position[1]

        if x is None or y is None:
            return None, None

        try:
            return float(x), float(y)
        except (TypeError, ValueError):
            return None, None

    @staticmethod
    def _is_valid_xy(x, y):
        if x is None or y is None:
            return False

        if np.isnan(x) or np.isnan(y):
            return False

        return True

    def collect_chunk_data(
        self,
        tracks,
        frame_offset,
        frame_width,
        frame_height,
        game_state_per_frame=None
    ):
        """
        Collects frame-by-frame player and ball data.

        Uses adjusted positions when available so camera motion does not distort
        the analytics as much.
        """
        for local_frame_num, player_track in enumerate(tracks["players"]):
            global_frame_num = frame_offset + local_frame_num

            frame_state = None
            if game_state_per_frame is not None and local_frame_num < len(game_state_per_frame):
                frame_state = game_state_per_frame[local_frame_num].get("state", None)

            ball_bbox = tracks["ball"][local_frame_num].get(1, {}).get("bbox", None)

            if ball_bbox is not None and len(ball_bbox) == 4:
                ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2

                self.ball_rows.append({
                    "frame": global_frame_num,
                    "state": frame_state,
                    "ball_x": ball_center_x,
                    "ball_y": ball_center_y,
                    "ball_x_relative": ball_center_x / frame_width,
                    "ball_y_relative": ball_center_y / frame_height
                })

            for player_id, player in player_track.items():
                raw_position = player.get("position", None)
                adjusted_position = player.get("position_adjusted", None)

                x_raw, y_raw = self._get_xy_from_position(raw_position)
                x_adj, y_adj = self._get_xy_from_position(adjusted_position)

                team = player.get("team", None)
                has_ball = player.get("has_ball", False)
                bbox = player.get("bbox", None)

                x_raw_relative = x_raw / frame_width if self._is_valid_xy(x_raw, y_raw) else None
                y_raw_relative = y_raw / frame_height if self._is_valid_xy(x_raw, y_raw) else None

                x_adj_relative = x_adj / frame_width if self._is_valid_xy(x_adj, y_adj) else None
                y_adj_relative = y_adj / frame_height if self._is_valid_xy(x_adj, y_adj) else None

                self.player_rows.append({
                    "frame": global_frame_num,
                    "state": frame_state,
                    "player_id": player_id,
                    "team": team,
                    "has_ball": has_ball,
                    "x": x_raw,
                    "y": y_raw,
                    "x_relative": x_raw_relative,
                    "y_relative": y_raw_relative,
                    "x_adjusted": x_adj,
                    "y_adjusted": y_adj,
                    "x_adjusted_relative": x_adj_relative,
                    "y_adjusted_relative": y_adj_relative,
                    "bbox": bbox
                })

                if has_ball and team is not None:
                    if self.last_player_with_ball is not None:
                        previous_player_id, previous_team = self.last_player_with_ball

                        if player_id != previous_player_id and team == previous_team:
                            self.pass_events.append({
                                "frame": global_frame_num,
                                "from_player": previous_player_id,
                                "to_player": player_id,
                                "team": team
                            })

                    self.last_player_with_ball = (player_id, team)

    def save_player_data(self):
        player_df = pd.DataFrame(self.player_rows)
        player_df.to_csv(
            os.path.join(self.output_dir, "player_data.csv"),
            index=False
        )

        ball_df = pd.DataFrame(self.ball_rows)
        ball_df.to_csv(
            os.path.join(self.output_dir, "ball_data.csv"),
            index=False
        )

        passes_df = pd.DataFrame(self.pass_events)
        passes_df.to_csv(
            os.path.join(self.output_dir, "passes.csv"),
            index=False
        )

    def save_summary(self, ball_control_history):
        ball_control_array = np.array(ball_control_history)

        team_1_frames = np.sum(ball_control_array == 1)
        team_2_frames = np.sum(ball_control_array == 2)
        total_control_frames = team_1_frames + team_2_frames

        if total_control_frames > 0:
            team_1_possession = team_1_frames / total_control_frames * 100
            team_2_possession = team_2_frames / total_control_frames * 100
        else:
            team_1_possession = 0
            team_2_possession = 0

        team_1_passes = len([p for p in self.pass_events if p["team"] == 1])
        team_2_passes = len([p for p in self.pass_events if p["team"] == 2])

        summary_df = pd.DataFrame([
            {
                "team": 1,
                "possession_percentage": team_1_possession,
                "estimated_passes": team_1_passes
            },
            {
                "team": 2,
                "possession_percentage": team_2_possession,
                "estimated_passes": team_2_passes
            }
        ])

        summary_df.to_csv(
            os.path.join(self.output_dir, "summary.csv"),
            index=False
        )

    def save_formations(self):
        player_df = pd.DataFrame(self.player_rows)

        if player_df.empty:
            return

        # Prefer adjusted positions for formations if available
        if "x_adjusted_relative" in player_df.columns and "y_adjusted_relative" in player_df.columns:
            player_df["heatmap_x"] = player_df["x_adjusted_relative"].where(
                player_df["x_adjusted_relative"].notna(),
                player_df["x_relative"]
            )
            player_df["heatmap_y"] = player_df["y_adjusted_relative"].where(
                player_df["y_adjusted_relative"].notna(),
                player_df["y_relative"]
            )
        else:
            player_df["heatmap_x"] = player_df["x_relative"]
            player_df["heatmap_y"] = player_df["y_relative"]

        player_df = player_df.dropna(subset=["team", "heatmap_x", "heatmap_y"])

        # Only keep plausible pitch coordinates
        player_df = player_df[
            (player_df["heatmap_x"] >= self.heatmap_min_relative) &
            (player_df["heatmap_x"] <= self.heatmap_max_relative) &
            (player_df["heatmap_y"] >= self.heatmap_min_relative) &
            (player_df["heatmap_y"] <= self.heatmap_max_relative)
        ]

        for team in [1, 2]:
            team_df = player_df[player_df["team"] == team]

            if team_df.empty:
                continue

            formation_df = team_df.groupby("player_id").agg({
                "heatmap_x": "mean",
                "heatmap_y": "mean",
                "frame": "count"
            }).reset_index()

            formation_df.rename(columns={"frame": "frames_detected"}, inplace=True)

            formation_df.to_csv(
                os.path.join(self.output_dir, f"formation_team_{team}.csv"),
                index=False
            )

    def _make_heatmap_array(self, team_df, x_col, y_col):
        x = team_df[x_col].to_numpy(dtype=np.float32)
        y = team_df[y_col].to_numpy(dtype=np.float32)

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if len(x) == 0:
            return None

        # Clip to normalized pitch bounds
        x = np.clip(x, self.heatmap_min_relative, self.heatmap_max_relative)
        y = np.clip(y, self.heatmap_min_relative, self.heatmap_max_relative)

        hist, x_edges, y_edges = np.histogram2d(
            x,
            y,
            bins=self.heatmap_bins,
            range=[
                [self.heatmap_min_relative, self.heatmap_max_relative],
                [self.heatmap_min_relative, self.heatmap_max_relative]
            ]
        )

        # Smooth the blocky bins
        hist = cv2.GaussianBlur(
            hist.astype(np.float32),
            (0, 0),
            sigmaX=self.heatmap_blur_sigma,
            sigmaY=self.heatmap_blur_sigma
        )

        return hist

    def save_heatmaps(self):
        player_df = pd.DataFrame(self.player_rows)

        if player_df.empty:
            return

        # Prefer adjusted positions if available
        if "x_adjusted_relative" in player_df.columns and "y_adjusted_relative" in player_df.columns:
            player_df["heatmap_x"] = player_df["x_adjusted_relative"].where(
                player_df["x_adjusted_relative"].notna(),
                player_df["x_relative"]
            )
            player_df["heatmap_y"] = player_df["y_adjusted_relative"].where(
                player_df["y_adjusted_relative"].notna(),
                player_df["y_relative"]
            )
        else:
            player_df["heatmap_x"] = player_df["x_relative"]
            player_df["heatmap_y"] = player_df["y_relative"]

        player_df = player_df.dropna(subset=["team", "heatmap_x", "heatmap_y"])

        # Optionally keep only live-play frames if game state was collected
        if self.heatmap_use_live_play_only and "state" in player_df.columns:
            live_mask = player_df["state"].isna() | (player_df["state"] == "LIVE_PLAY")
            player_df = player_df[live_mask]

        # Remove impossible values / edge garbage
        player_df = player_df[
            (player_df["heatmap_x"] >= self.heatmap_min_relative) &
            (player_df["heatmap_x"] <= self.heatmap_max_relative) &
            (player_df["heatmap_y"] >= self.heatmap_min_relative) &
            (player_df["heatmap_y"] <= self.heatmap_max_relative)
        ]

        for team in [1, 2]:
            team_df = player_df[player_df["team"] == team]

            if team_df.empty:
                continue

            heatmap = self._make_heatmap_array(team_df, "heatmap_x", "heatmap_y")

            if heatmap is None:
                continue

            # Normalize for display
            max_value = np.max(heatmap)
            if max_value > 0:
                heatmap = heatmap / max_value

            plt.figure(figsize=(10, 6))
            plt.imshow(
                heatmap.T,
                origin="upper",
                extent=[
                    self.heatmap_min_relative,
                    self.heatmap_max_relative,
                    self.heatmap_max_relative,
                    self.heatmap_min_relative
                ],
                cmap="plasma",
                aspect="auto",
                interpolation="bilinear"
            )
            plt.title(f"Team {team} Relative Position Heatmap")
            plt.xlabel("Relative Field X")
            plt.ylabel("Relative Field Y")
            plt.colorbar(label="Normalized Position Density")
            plt.tight_layout()

            plt.savefig(
                os.path.join(self.output_dir, f"team_{team}_heatmap.png"),
                dpi=200
            )
            plt.close()

    def save_all(self, ball_control_history):
        self.save_player_data()
        self.save_summary(ball_control_history)
        self.save_formations()
        self.save_heatmaps()
