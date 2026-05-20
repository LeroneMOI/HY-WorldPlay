import numpy as np
import json


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class CameraState:
    def __init__(self):
        self.T = np.eye(4)
        self.intrinsic = [
            [969.6969696969696, 0.0, 960.0],
            [0.0, 969.6969696969696, 540.0],
            [0.0, 0.0, 1.0],
        ]
        self.last_c2w = None

    def reset(self):
        self.T = np.eye(4)
        self.last_c2w = None

    def save(self):
        state = {
            "T": self.T.tolist(),
            "intrinsic": self.intrinsic,
            "last_c2w": self.last_c2w.tolist() if self.last_c2w is not None else None,
        }
        return state

    def restore(self, state):
        self.T = np.array(state["T"])
        self.intrinsic = state["intrinsic"]
        self.last_c2w = np.array(state["last_c2w"]) if state["last_c2w"] is not None else None


_global_camera_state = CameraState()


def generate_camera_trajectory_local(motions, camera_state=None):
    """
    motions: list of dict
             {"forward": 1.0}, {"yaw": np.pi/2}, {"pitch": np.pi/6}, {"right": 1.0}
             - forward: Translation (Forward or Backward)
             - yaw:   Rotate (Left or Right)
             - pitch: Rotate (Up or Down)
             - right: Translation (Right or Left)
             - third_yaw: Third Perspective Rotate (Left or Right)

    camera_state: CameraState instance for persistent camera transform across rounds.
                  If None, uses global camera state.
    """
    if camera_state is None:
        camera_state = _global_camera_state

    T = camera_state.T.copy()
    poses = [T.copy()]

    for move in motions:
        if "yaw" in move:
            R = rot_y(move["yaw"])
            T[:3, :3] = T[:3, :3] @ R

        if "pitch" in move:
            R = rot_x(move["pitch"])
            T[:3, :3] = T[:3, :3] @ R

        forward = move.get("forward", 0.0)
        if forward != 0:
            local_t = np.array([0, 0, forward])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        right = move.get("right", 0.0)
        if right != 0:
            local_t = np.array([right, 0, 0])
            world_t = T[:3, :3] @ local_t
            T[:3, 3] += world_t

        third_yaw = move.get("third_yaw", 0.0)
        if third_yaw != 0:
            theta = -third_yaw
            C = np.array([[1, 0.0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1.0], [0, 0, 0, 1]])
            c_origin = C.copy()
            R_y = np.array(
                [
                    [np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                ]
            )
            C[:3, :3] = C[:3, :3] @ R_y
            C[:3, 3] = R_y @ C[:3, 3]
            c_inv = np.linalg.inv(c_origin)
            c_relative = c_inv @ C
            T = T @ c_relative

        poses.append(T.copy())

    camera_state.T = T
    camera_state.last_c2w = T.copy()

    return poses


if __name__ == "__main__":
    intrinsic = [
        [969.6969696969696, 0.0, 960.0],
        [0.0, 969.6969696969696, 540.0],
        [0.0, 0.0, 1.0],
    ]

    motions = []
    for i in range(15):
        motions.append({"forward": 0.08})

    for i in range(16):
        motions.append({"yaw": np.deg2rad(3)})

    poses = generate_camera_trajectory_local(motions)
    custom_c2w = {}
    for i, p in enumerate(poses):
        custom_c2w[str(i)] = {"extrinsic": p.tolist(), "K": intrinsic}
        json.dump(
            custom_c2w,
            open("./assets/pose/pose.json", "w"),
            indent=4,
            ensure_ascii=False,
        )
