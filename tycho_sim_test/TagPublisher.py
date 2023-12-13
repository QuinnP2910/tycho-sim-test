import rospy
import tf2_ros
from geometry_msgs.msg import PointStamped, PoseStamped, TransformStamped

def main():
    rospy.init_node('tag_publisher')
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    publisher = rospy.Publisher('tag_1_translation', PoseStamped, queue_size=10)

    point_in_frame = PointStamped()
    point_in_frame.header.frame_id = 'head_camera'
    point_in_frame.point.x = 0.0
    point_in_frame.point.y = 0.0
    point_in_frame.point.z = 0.0

    rate = rospy.Rate(10.0)

    while not rospy.is_shutdown():
        try:
            transform = tfBuffer.lookup_transform('head_camera', 'tag_1', rospy.Time())

            pose_msg = PoseStamped()
            pose_msg.header.frame_id = 'head_camera'
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.pose.position = transform.transform.translation
            pose_msg.pose.orientation = transform.transform.rotation
            publisher.publish(pose_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rate.sleep()
            continue

        rate.sleep()

if __name__ == '__main__':
    main()