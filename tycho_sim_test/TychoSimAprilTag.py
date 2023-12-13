import numpy as np, rospy
from threading import Lock
import rospy
import math
from time import time

# from scipy.spatial.transform import Rotation as R

from mocap_optitrack.msg import PointArray
from TychoSim import TychoSim
from geometry_msgs.msg import PoseStamped
APRILTAG_TRANSLATION_TOPIC_NAME = '/tag_1_translation'
apriltag_translation = None

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z] # in radians

def _sim_teleop():
    global apriltag_translation
    def callback(data):
        global apriltag_translation
        apriltag_translation = data

    rospy.Subscriber(APRILTAG_TRANSLATION_TOPIC_NAME, PoseStamped, callback, queue_size=10)

    print("Starting sim")
    sim = TychoSim(([-1.69926117, 1.91009097, 2.09709026, -0.09968156, 1.5817661, 0.07704814, -0.37601018]))
    sim.run_simulation()

    while (apriltag_translation is None):
        pass

    chop_translation = apriltag_translation.pose.position
    chop_translation_array = np.array([chop_translation.z, -chop_translation.x, -chop_translation.y])
    chop_orientation = apriltag_translation.pose.orientation
    chop_orientation_euler = euler_from_quaternion(chop_orientation.x, chop_orientation.y, chop_orientation.z, chop_orientation.w)
    rotated_chop_orientation_euler = np.array([-chop_orientation_euler[2], chop_orientation_euler[0], -chop_orientation_euler[1]])

    sim.init_leader_position(chop_translation_array, rotated_chop_orientation_euler)

    while (True):
        chop_translation = apriltag_translation.pose.position
        chop_translation_array = np.array([chop_translation.z, -chop_translation.x, -chop_translation.y])
        chop_orientation = apriltag_translation.pose.orientation
        chop_orientation_euler = euler_from_quaternion(chop_orientation.x, chop_orientation.y, chop_orientation.z, chop_orientation.w)
        # print("Chopsticks orientation pre-rot: " + str(chop_orientation_euler) + "\r\n")
        rotated_chop_orientation_euler = np.array([-chop_orientation_euler[2], chop_orientation_euler[0], -chop_orientation_euler[1]])
        # print("Chopsticks orientation: " + str(rotated_chop_orientation_euler) + "\r\n")

        sim.set_leader_position(chop_translation_array, rotated_chop_orientation_euler)

        # Obtain joint positions from Mujoco simulation
        mujoco_joint_positions = sim.get_joint_positions()
        mujoco_joint_positions[6] = 0.0

        joint_target = mujoco_joint_positions

        # print(str(joint_target) + "\r\n")

if __name__ == '__main__':
    rospy.init_node("TychoSimAprilTag")
    _sim_teleop()
    rospy.spin()