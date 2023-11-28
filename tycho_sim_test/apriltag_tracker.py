import cv2
import numpy as np

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

        cameraMatrix = np.array([[1.38177620e+03, 0.00000000e+00, 5.68619767e+02],
                                 [0.00000000e+00, 1.44635760e+03, 3.38215946e+02],
                                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        distCoeffs = np.array([-2.59102674e-01, 3.74137215e+00, -9.23851828e-04, 7.21012570e-03,
                               -2.99522420e+01])

        # Estimate pose
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1.0, cameraMatrix=cameraMatrix,
                                                            distCoeffs=distCoeffs)

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Extract position (x, y, z)
        x, y, z = tvec[0][0]

        # Extract euler angles (yaw, pitch, roll)
        angles = cv2.RQDecomp3x3(R)[0]
        yaw, pitch, roll = angles[0], angles[1], angles[2]

        return tag_id, x, y, z, yaw, pitch, roll

    return None


# Main function
def main(sim):
    # Open video capture (0 for default camera)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Detect and track AprilTag
        result = track_apriltag(frame)

        if result:
            tag_id, x, y, z, yaw, pitch, roll = result
            simulation_instance.set_leader_position([x, y, z], [yaw, pitch, roll])
            print(f"AprilTag ID: {tag_id}")
            print(f"Position: ({x:.2f}, {y:.2f}, {z:.2f})")
            print(f"Euler Angles: Yaw={yaw:.2f}, Pitch={pitch:.2f}, Roll={roll:.2f}")
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
    simulation_instance = TychoSim(
        [-2.386569349825415, 0.5591684504416656, 0.3236793467739023, -1.78454031627861, 1.5790132769886849,
         -0.8286323184826947, 0.00806077238186087])
    simulation_instance.run_simulation()
    main(simulation_instance)
