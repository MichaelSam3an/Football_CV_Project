import sys
sys.path.append('../')

from utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner():

    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(self, players, ball_bbox):

        ball_position = get_center_of_bbox(ball_bbox)

        minimum_distance = 99999
        assigned_player = -1

        ball_x, ball_y = ball_position

        for player_id, player in players.items():

            player_bbox = player['bbox']

            x1, y1, x2, y2 = player_bbox

            # ==================================
            # FULL BODY POSSESSION ZONE
            # ==================================

            padding_x = 25
            padding_y = 40

            inside_player_zone = (
                ball_x >= x1 - padding_x and
                ball_x <= x2 + padding_x and
                ball_y >= y1 - padding_y and
                ball_y <= y2 + padding_y
            )

            # ==================================
            # FOOT DISTANCE (fallback)
            # ==================================

            distance_left = measure_distance(
                (x1, y2),
                ball_position
            )

            distance_right = measure_distance(
                (x2, y2),
                ball_position
            )

            distance = min(distance_left, distance_right)

            # ==================================
            # COMBINED LOGIC
            # ==================================

            if inside_player_zone or distance < self.max_player_ball_distance:

                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
