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

    def collect_chunk_data(
        self,
        tracks,
        frame_offset,
        frame_width,
        frame_height
    ):
        for local_frame_num, player_track in enumerate(tracks["players"]):
            global_frame_num = frame_offset + local_frame_num

            ball_bbox = tracks["ball"][local_frame_num].get(1, {}).get("bbox", None)

            if ball_bbox is not None and len(ball_bbox) == 4:
                ball_center_x = (ball_bbox[0] + ball_bbox[2]) / 2
                ball_center_y = (ball_bbox[1] + ball_bbox[3]) / 2

                self.ball_rows.append({
                    "frame": global_frame_num,
                    "ball_x": ball_center_x,
                    "ball_y": ball_center_y,
                    "ball_x_relative": ball_center_x / frame_width,
                    "ball_y_relative": ball_center_y / frame_height
                })

            for player_id, player in player_track.items():
                position = player.get("position", (None, None))
                team = player.get("team", None)
                has_ball = player.get("has_ball", False)
                bbox = player.get("bbox", None)

                x = position[0] if position else None
                y = position[1] if position else None

                x_relative = x / frame_width if x is not None else None
                y_relative = y / frame_height if y is not None else None

                self.player_rows.append({
                    "frame": global_frame_num,
                    "player_id": player_id,
                    "team": team,
                    "has_ball": has_ball,
                    "x": x,
                    "y": y,
                    "x_relative": x_relative,
                    "y_relative": y_relative,
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

        player_df = player_df.dropna(subset=["team", "x_relative", "y_relative"])

        for team in [1, 2]:
            team_df = player_df[player_df["team"] == team]

            if team_df.empty:
                continue

            formation_df = team_df.groupby("player_id").agg({
                "x_relative": "mean",
                "y_relative": "mean",
                "frame": "count"
            }).reset_index()

            formation_df.rename(columns={"frame": "frames_detected"}, inplace=True)

            formation_df.to_csv(
                os.path.join(self.output_dir, f"formation_team_{team}.csv"),
                index=False
            )

    def save_heatmaps(self):
        player_df = pd.DataFrame(self.player_rows)

        if player_df.empty:
            return

        player_df = player_df.dropna(subset=["team", "x_relative", "y_relative"])

        for team in [1, 2]:
            team_df = player_df[player_df["team"] == team]

            if team_df.empty:
                continue

            plt.figure(figsize=(10, 6))
            plt.hist2d(
                team_df["x_relative"],
                team_df["y_relative"],
                bins=50
            )
            plt.gca().invert_yaxis()
            plt.title(f"Team {team} Relative Position Heatmap")
            plt.xlabel("Relative Field X")
            plt.ylabel("Relative Field Y")
            plt.colorbar(label="Player Position Frequency")
            plt.tight_layout()

            plt.savefig(
                os.path.join(self.output_dir, f"team_{team}_heatmap.png")
            )

            plt.close()

    def save_all(self, ball_control_history):
        self.save_player_data()
        self.save_summary(ball_control_history)
        self.save_formations()
        self.save_heatmaps()