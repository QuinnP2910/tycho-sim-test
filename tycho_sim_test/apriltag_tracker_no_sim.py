import cv2
import numpy as np
import math


# Function to detect and track AprilTag
def track_apriltag(image):
    # Create AprilTag dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)

    # Create AprilTag parameters
    parameters = cv2.aruco.DetectorParameters()

    # Detect AprilTags
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    print(corners)

    if ids is not None:
        # Assuming you have a single AprilTag in the scene
        tag_id = ids[0][0]

        cameraMatrix = np.array([[959.30876891, 0., 590.29902055],
                                 [0., 976.07955436, 467.10804575],
                                 [0., 0., 1., ]])

        distCoeffs = np.array([-0.14018771, 0.17704573, 0.02988585, -0.01282493, -0.10035589])

        # Estimate pose
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix=cameraMatrix,
                                                            distCoeffs=distCoeffs)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        camera_rot = np.radians(45)
        rotation_matrix_45_deg = np.array([[np.cos(camera_rot), 0, np.sin(camera_rot)],
                                           [0, 1, 0],
                                           [-np.sin(camera_rot), 0, np.cos(camera_rot)]])
        R = np.dot(rotation_matrix_45_deg, R)

        # Extract position (x, y, z)
        x, y, z = tvec[0][0]

        # Extract euler angles (yaw, pitch, roll)
        angles = cv2.RQDecomp3x3(R)[0]
        yaw, pitch, roll = math.radians(angles[0]), math.radians(angles[1]), math.radians(angles[2])

        return tag_id, x, y, z, yaw, pitch, roll

    return None


# Main function
def main():
    # Open video capture (0 for default camera)
    print("Connecting to cam...")
    cap = cv2.VideoCapture(2)
    print("Connected!")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect and track AprilTag
        result = track_apriltag(frame)

        if result:
            tag_id, x, y, z, yaw, pitch, roll = result
            print(f"AprilTag ID: {tag_id}")
            print(f"Position: ({x:.5f}, {y:.5f}, {z:.5f})")
            print(f"Euler Angles: Yaw={yaw:.5f}, Pitch={pitch:.5f}, Roll={roll:.5f}")
            print()

        # Display the frame
        cv2.imshow('AprilTag Tracking', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
