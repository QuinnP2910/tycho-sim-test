import cv2
import apriltag
import math

def get_euler_angles(rotation_matrix):
    pitch = -math.asin(rotation_matrix[2, 0])
    roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    pitch = math.degrees(pitch)
    roll = math.degrees(roll)
    yaw = math.degrees(yaw)

    return yaw, pitch, roll

def main():
    cap = cv2.VideoCapture(-1)

    # Create an AprilTag detector
    detector = apriltag.Detector()

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect AprilTags in the image
        detections, dimg = detector.detect(gray, return_image=True)

        if detections:
            for detection in detections:
                # Extract the tag ID and pose information
                tag_id = detection.tag_id
                pose_rvec, pose_tvec, _ = detector.detection_pose(detection, camera_params=(640, 480, 640, 480))

                # Convert rotation vector to rotation matrix
                pose_rmat, _ = cv2.Rodrigues(pose_rvec)

                # Extract Euler angles from the rotation matrix
                yaw, pitch, roll = get_euler_angles(pose_rmat)

                # Display the tag ID and pose information
                print(f"Tag ID: {tag_id}")
                print(f"Pose (x, y, z): {pose_tvec.flatten()}")
                print(f"Euler Angles (yaw, pitch, roll): {yaw}, {pitch}, {roll}")

                # Draw the AprilTag on the frame
                detector.detection_draw(detection, frame)

        cv2.imshow('AprilTag Tracking', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
