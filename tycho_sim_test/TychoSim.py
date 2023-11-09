import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('./assets/hebi.xml')
d = mujoco.MjData(m)
# d.geom_xmat[m.geom('leader_stick').id] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
mujoco.mj_step(m, d)

# Get the site index for 'leader_fixed_chop_tip_no_rot'
leader_site_idx = m.site('leader_fixed_chop_tip_no_rot').id

x_actuator_idx = m.actuator('x').id
y_actuator_idx = m.actuator('y').id
z_actuator_idx = m.actuator('z').id
rx_actuator_idx = m.actuator('rx').id
ry_actuator_idx = m.actuator('ry').id
rz_actuator_idx = m.actuator('rz').id

leader_starting_pos_x = d.site_xpos[leader_site_idx][0]
leader_starting_pos_y = d.site_xpos[leader_site_idx][1]
leader_starting_pos_z = d.site_xpos[leader_site_idx][2]
leader_starting_mat = d.site_xmat[leader_site_idx]

def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
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

def calculate_angle_difference(angle1, angle2):
    diff = angle2 - angle1
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    return diff

leader_starting_pos_rx = 3.14
leader_starting_pos_ry = 0
leader_starting_pos_rz = 3.14

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # mj_step can be replaced with code that also evaluates
        # a policy and applies a control signal before stepping the physics.
        mujoco.mj_step(m, d)

        leader_pos = d.site_xpos[leader_site_idx]
        leader_euler = mat2euler(d.site_xmat[leader_site_idx])

        # print(leader_starting_pos_x)
        d.ctrl[x_actuator_idx] = -(leader_pos[0] - leader_starting_pos_x)
        d.ctrl[y_actuator_idx] = leader_pos[1] - leader_starting_pos_y
        d.ctrl[z_actuator_idx] = -(leader_pos[2] - leader_starting_pos_z)

        ALLOWABLE_ROT_MOVEMENT = 1.0
        d.ctrl[rx_actuator_idx] = calculate_angle_difference(leader_euler[0], leader_starting_pos_rx)
        if d.ctrl[rx_actuator_idx] < -ALLOWABLE_ROT_MOVEMENT or d.ctrl[rx_actuator_idx] > ALLOWABLE_ROT_MOVEMENT:
            d.ctrl[rx_actuator_idx] = 0.0
        d.ctrl[ry_actuator_idx] = calculate_angle_difference(leader_euler[1], leader_starting_pos_ry)
        if d.ctrl[ry_actuator_idx] < -ALLOWABLE_ROT_MOVEMENT or d.ctrl[ry_actuator_idx] > ALLOWABLE_ROT_MOVEMENT:
            d.ctrl[ry_actuator_idx] = 0.0
        d.ctrl[rz_actuator_idx] = -calculate_angle_difference(leader_euler[2], leader_starting_pos_rz)
        if d.ctrl[rz_actuator_idx] < -ALLOWABLE_ROT_MOVEMENT or d.ctrl[rz_actuator_idx] > ALLOWABLE_ROT_MOVEMENT:
            d.ctrl[rz_actuator_idx] = 0.0

        # print(d.ctrl[rz_actuator_idx])
        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()

