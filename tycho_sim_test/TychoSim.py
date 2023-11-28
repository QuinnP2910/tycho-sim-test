import mujoco
import mujoco.viewer
import numpy as np

class TychoSim:
    def __init__(self, joint_positions, model_path='./assets/hebi.xml'):
        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)

        self.set_joint_positions(joint_positions)
        mujoco.mj_step(self.m, self.d)

        self.viewer = None  # Viewer will be initialized in the run_simulation method

        # Define actuator indices
        self.x_actuator_idx = self.m.actuator('x').id
        self.y_actuator_idx = self.m.actuator('y').id
        self.z_actuator_idx = self.m.actuator('z').id
        self.rx_actuator_idx = self.m.actuator('rx').id
        self.ry_actuator_idx = self.m.actuator('ry').id
        self.rz_actuator_idx = self.m.actuator('rz').id

        # Initial leader position and orientation
        self.leader_site_idx = self.m.site('leader_fixed_chop_tip_no_rot').id
        self.leader_starting_pos_x = self.d.site_xpos[self.leader_site_idx][0]
        self.leader_starting_pos_y = self.d.site_xpos[self.leader_site_idx][1]
        self.leader_starting_pos_z = self.d.site_xpos[self.leader_site_idx][2]
        self.leader_starting_mat = self.d.site_xmat[self.leader_site_idx]

        # Initial orientation angles
        self.leader_starting_pos_rx = 3.14
        self.leader_starting_pos_ry = 0
        self.leader_starting_pos_rz = 3.14

    def mat2euler(self, mat):
        mat = mat.reshape(3, 3)
        mat = np.asarray(mat, dtype=np.float64)
        assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

        cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
        condition = True
        euler = np.empty(mat.shape[:-1], dtype=np.float64)
        euler[..., 2] = np.where(condition,
                                 -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                                 -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
        euler[..., 1] = np.where(condition,
                                 -np.arctan2(-mat[..., 0, 2], cy),
                                 -np.arctan2(-mat[..., 0, 2], cy))
        euler[..., 0] = np.where(condition,
                                 -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                                 0.0)
        return euler

    def calculate_angle_difference(self, angle1, angle2):
        diff = angle2 - angle1
        if diff > np.pi:
            diff -= 2 * np.pi
        elif diff < -np.pi:
            diff += 2 * np.pi
        return diff

    def run_simulation(self):
        with mujoco.viewer.launch_passive(self.m, self.d) as self.viewer:
            while self.viewer.is_running():
                self.update_simulation_state()

    def get_joint_positions(self):
        joint_positions = []

        joint_positions.append(self.d.joint("HEBI/base/X8_9").qpos[0])
        joint_positions.append(self.d.joint("HEBI/shoulder/X8_16").qpos[0])
        joint_positions.append(self.d.joint("HEBI/elbow/X8_9").qpos[0])
        joint_positions.append(self.d.joint("HEBI/wrist1/X5_1").qpos[0])
        joint_positions.append(self.d.joint("HEBI/wrist2/X5_1").qpos[0])
        joint_positions.append(self.d.joint("HEBI/wrist3/X5_1").qpos[0])
        joint_positions.append(self.d.joint("HEBI/chopstick/X5_1").qpos[0])

        return joint_positions

    def set_joint_positions(self, joint_positions):
        self.d.joint("HEBI/base/X8_9").qpos[0] = joint_positions[0]
        self.d.joint("HEBI/shoulder/X8_16").qpos[0] = joint_positions[1]
        self.d.joint("HEBI/elbow/X8_9").qpos[0] = joint_positions[2]
        self.d.joint("HEBI/wrist1/X5_1").qpos[0] = joint_positions[3]
        self.d.joint("HEBI/wrist2/X5_1").qpos[0] = joint_positions[4]
        self.d.joint("HEBI/wrist3/X5_1").qpos[0] = joint_positions[5]
        self.d.joint("HEBI/chopstick/X5_1").qpos[0] = joint_positions[6]

    def update_simulation_state(self):
        print(self.get_joint_positions())
        mujoco.mj_step(self.m, self.d)

        leader_pos = self.d.site_xpos[self.leader_site_idx]
        leader_euler = self.mat2euler(self.d.site_xmat[self.leader_site_idx])

        # Update control signals
        self.d.ctrl[self.x_actuator_idx] = -(leader_pos[0] - self.leader_starting_pos_x)
        self.d.ctrl[self.y_actuator_idx] = leader_pos[1] - self.leader_starting_pos_y
        self.d.ctrl[self.z_actuator_idx] = -(leader_pos[2] - self.leader_starting_pos_z)

        ALLOWABLE_ROT_MOVEMENT = 1.0
        self.d.ctrl[self.rx_actuator_idx] = self.calculate_angle_difference(
            leader_euler[0], self.leader_starting_pos_rx
        )
        if (
            self.d.ctrl[self.rx_actuator_idx] < -ALLOWABLE_ROT_MOVEMENT
            or self.d.ctrl[self.rx_actuator_idx] > ALLOWABLE_ROT_MOVEMENT
        ):
            self.d.ctrl[self.rx_actuator_idx] = 0.0
        self.d.ctrl[self.ry_actuator_idx] = self.calculate_angle_difference(
            leader_euler[1], self.leader_starting_pos_ry
        )
        if (
            self.d.ctrl[self.ry_actuator_idx] < -ALLOWABLE_ROT_MOVEMENT
            or self.d.ctrl[self.ry_actuator_idx] > ALLOWABLE_ROT_MOVEMENT
        ):
            self.d.ctrl[self.ry_actuator_idx] = 0.0
        self.d.ctrl[self.rz_actuator_idx] = -self.calculate_angle_difference(
            leader_euler[2], self.leader_starting_pos_rz
        )
        if (
            self.d.ctrl[self.rz_actuator_idx] < -ALLOWABLE_ROT_MOVEMENT
            or self.d.ctrl[self.rz_actuator_idx] > ALLOWABLE_ROT_MOVEMENT
        ):
            self.d.ctrl[self.rz_actuator_idx] = 0.0

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        self.viewer.sync()

if __name__ == '__main__':
    simulation_instance = TychoSim([-2.386569349825415, 0.5591684504416656, 0.3236793467739023, -1.78454031627861, 1.5790132769886849, -0.8286323184826947, 0.00806077238186087])
    simulation_instance.run_simulation()