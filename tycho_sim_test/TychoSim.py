import mujoco_py
import glfw

if __name__ == '__main__':
    # Load the model
    model = mujoco_py.load_model_from_path("assets/hebi.xml")

    # Create the simulation environment
    sim = mujoco_py.MjSim(model)

    # Set the desired position of the secondary joint(s)
    desired_position = 0.5
    sim.data.qpos[9] = desired_position

    # Initialize GLFW and create a rendering context
    glfw.init()
    mujoco_py.builder.MujocoPyOpenGLContext(offscreen=True)

    try:
        while True:
            # Update simulation state (e.g., set joint positions)
            # sim.data.qpos[:] = ...  # Update joint positions if necessary

            # Step the simulation
            sim.step()

            # Render the simulation
            viewer = mujoco_py.MjRenderContextOffscreen(sim, -1)
            viewer.render(640, 480)  # Set the desired window size

            # Check for user input events (e.g., keyboard or mouse events)
            glfw.poll_events()

            # Break the loop if the user closes the window
            if glfw.window_should_close(viewer.window):
                break

    finally:
        # Clean up GLFW
        glfw.terminate()