import numpy as np
import robosuite as suite
import time
import csv
import os

# Create environment instance with default controller
env = suite.make(
    env_name="Stack",
    robots="UR5e",
    horizon=1200,
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,  # Enable object observations
    placement_initializer=None
)

# Reset the environment
obs = env.reset()

# Get the locations of all the blocks
cubeA_center = obs["cubeA_pos"][0:1]   # e.g. array([x, y, z])
cubeB_center = obs["cubeB_pos"][0:1]
cubeC_center = obs["cubeC_pos"][0:1]
cubeD_center = obs["cubeD_pos"][0:1]
cubeE_center = obs["cubeE_pos"][0:1]
cubeF_center = obs["cubeF_pos"][0:1]
print("Red block at", cubeA_center)
print("Green block at", cubeB_center)
print("Blue block at", cubeC_center)
print("Yellow block at", cubeD_center)
print("Orange block at", cubeE_center)
print("Purple block at", cubeF_center)

# Print observation keys and action dimensions for debugging
print("Observation keys:", obs.keys())
print("Action dimensions:", env.action_dim)
print("Action space:", env.action_spec)

# Read trajectory from CSV file
def read_trajectory_from_csv(file_path):
    """Read trajectory points from a CSV file."""
    trajectory = []
    descriptions = []
    
    with open(file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        header = next(csv_reader)  # Skip header row
        
        for row in csv_reader:
            if len(row) >= 3:
                # Extract x, y, z coordinates
                point = [float(row[0]), float(row[1]), float(row[2])]
                trajectory.append(point)
                
                # Extract description if available
                if len(row) >= 4:
                    descriptions.append(row[3])
                else:
                    descriptions.append("")
    
    return np.array(trajectory), descriptions

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "mpc_trajectory.csv")

# Load the trajectory
trajectory, descriptions = read_trajectory_from_csv(csv_path)
print(f"Loaded trajectory with {len(trajectory)} points from {csv_path}")
for i, (point, desc) in enumerate(zip(trajectory, descriptions)):
    print(f"  Point {i}: {point} - {desc}")

# Parameters for trajectory following
point_idx = 0                # Current trajectory point index
point_threshold = 0.08       # Distance threshold to consider a point reached (increased for tolerance)
max_steps_per_point = 150    # Maximum steps before moving to next point
current_point_steps = 0      # Steps spent on current point
kp = 2.0                    # Proportional gain (reduced for smoother motion)

print("\nStarting trajectory following...")
print(f"Trajectory has {len(trajectory)} points")
print("Will follow points in sequence, considering a point reached when end-effector is within", 
      f"{point_threshold} distance or after {max_steps_per_point} steps")

# Define orientation to face the user (assumed to be in front of the robot)
# For a Panda robot, this orients the end-effector to point toward negative y-axis
# Format: [rotation around x, rotation around y, rotation around z] in axis-angle format
facing_user_orientation = np.array([0, 0, 0])

# Main control loop
for i in range(5000):  # Increased to give more time
    # Get current end-effector position
    ee_pos = obs["robot0_eef_pos"]
    
    # Get current target point from trajectory
    target_pos = trajectory[point_idx]
    description = descriptions[point_idx]
    
    # Calculate distance to target
    distance = np.linalg.norm(ee_pos - target_pos)
    
    # Check if we've reached the current point or spent too much time trying
    if distance < point_threshold or current_point_steps >= max_steps_per_point:
        point_idx = (point_idx + 1) % len(trajectory)
        current_point_steps = 0
        print(f"Moving to next trajectory point {point_idx}: {trajectory[point_idx]} - {descriptions[point_idx]}")
    else:
        current_point_steps += 1
    
    # Calculate action: direction vector to target scaled by gain
    delta = target_pos - ee_pos
    position_action = kp * delta
    
    # Clip action to reasonable bounds
    position_action = np.clip(position_action, -1.0, 1.0)
    
    # Check action space to determine correct action format
    action_dim = env.action_dim
    
    # Create full action with appropriate dimensions
    action = np.concatenate([position_action, facing_user_orientation, [0.0]])
        
    # Take step in environment
    obs, reward, done, info = env.step(action)
    
    # Render the environment
    env.render()
    
    # Print progress occasionally
    if i % 100 == 0:
        print(f"Step {i}, Point {point_idx}/{len(trajectory)-1}, Distance: {distance:.4f}")
        print(f"  Current position: {ee_pos}")
        print(f"  Target position: {target_pos} - {description}")
        print(f"  Action: {position_action}")
    
    # Small delay for visualization
    time.sleep(0.01)
    
    if done:
        print("Environment signaled done!")
        break

print("Trajectory following completed!")
env.close()