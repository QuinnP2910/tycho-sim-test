import cv2
import numpy as np
import math

from TychoSim import TychoSim


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

        cameraMatrix = np.array([[1.02388096e+03, 0.00000000e+00, 5.92969843e+02]
, [0.00000000e+00, 1.01675390e+03, 2.99690700e+02]
, [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        distCoeffs = np.array([-0.07735032, 0.22902652, -0.00891713, -0.01716781, -0.25817705])

        # Estimate pose
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix=cameraMatrix,
                                                            distCoeffs=distCoeffs)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        # camera_rot = np.radians(45)
        # rotation_matrix_45_deg = np.array([[np.cos(camera_rot), 0, np.sin(camera_rot)],
        #                                    [0, 1, 0],
        #                                    [-np.sin(camera_rot), 0, np.cos(camera_rot)]])
        # R = np.dot(rotation_matrix_45_deg, R)


        # Extract position (x, y, z)
        x, y, z = tvec[0][0]

        # Extract euler angles (yaw, pitch, roll)
        angles = cv2.RQDecomp3x3(R)[0]
        yaw, pitch, roll = math.radians(angles[0]), math.radians(angles[1]), math.radians(angles[2])

        return tag_id, x, y, z, yaw, pitch, roll

    return None


# Main function
def main(sim):
    # Open video capture (0 for default camera)
    cap = cv2.VideoCapture(2)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect and track AprilTag
        result = track_apriltag(frame)

        if result:
            tag_id, x, y, z, yaw, pitch, roll = result
            simulation_instance.set_leader_position([z, -x, -y], [-roll, -yaw, pitch])
            print(f"AprilTag ID: {tag_id}")
            print(f"Position: ({x:.5f}, {y:.5f}, {z:.5f})")
            print(f"Euler Angles: Yaw={yaw:.5f}, Pitch={pitch:.5f}, Roll={roll:.5f}")
            print()

        # Display the frame
        cv2.imshow('AprilTag Tracking', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            if result:
                simulation_instance.init_leader_position([z, -x, -y], [-roll, -yaw, pitch])
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    simulation_instance = TychoSim(
        [-2.386569349825415, 0.5591684504416656, 0.3236793467739023, -1.78454031627861, 1.5790132769886849,
         -0.8286323184826947, 0.00806077238186087])
    simulation_instance.run_simulation()
    main(simulation_instance)
