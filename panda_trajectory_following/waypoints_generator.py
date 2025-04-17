import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


def generate_waypoints_mpc(
    start_point: np.ndarray,
    end_point: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    dt: float = 0.1,
    horizon: int = 20,
    num_waypoints: int = 50,
    max_velocity: float = 2,
    max_acceleration: float = 1,
    obstacle_buffer: float = 0.2,
    Q_goal: float = 100.0,
    R_input: float = 1.0,
    Q_obstacle: float = 100.0,
    solver_verbose: bool = False,
    Debug: bool = True,  # Set to True to see progress
    goal_tolerance: float = 0.02
) -> np.ndarray:
    """
    Generate waypoints from start to end point using MPC, avoiding obstacles in the x-y plane.
    
    Args:
        start_point: Starting point [x, y]
        end_point: Goal point [x, y]
        obstacles: List of obstacles specified as [(position, radius)], where position is [x, y]
        dt: Time step for the MPC model
        horizon: Prediction horizon for MPC
        num_waypoints: Number of waypoints to generate
        max_velocity: Maximum velocity constraint
        max_acceleration: Maximum acceleration constraint
        obstacle_buffer: Additional safety buffer around obstacles
        Q_goal: Weight for goal reaching objective
        R_input: Weight for control input regularization
        Q_obstacle: Weight for obstacle avoidance
        solver_verbose: Whether to print solver details for debugging
        Debug: Whether to print debug information
        goal_tolerance: Distance tolerance for reaching the goal
    Returns:
        np.ndarray: Array of waypoints with shape (num_waypoints, 2) where each row is [x, y]
    """
    # Convert inputs to numpy arrays
    start_point = np.array(start_point, dtype=float).reshape(2)
    end_point = np.array(end_point, dtype=float).reshape(2)
    
    # Distance between start and end
    total_distance = np.linalg.norm(end_point - start_point)
    
    # State dimension (position and velocity in x-y: [x, y, vx, vy])
    state_dim = 4
    
    # Control input dimension (acceleration in x-y: [ax, ay])
    control_dim = 2
    
    # Define MPC system matrices (double integrator model)
    A = np.block([
        [np.eye(2), dt * np.eye(2)],
        [np.zeros((2, 2)), np.eye(2)]
    ])
    
    B = np.block([
        [0.5 * dt**2 * np.eye(2)],
        [dt * np.eye(2)]
    ])
    
    # Initialize state at the start point with zero velocity
    current_state = np.zeros(state_dim)
    current_state[:2] = start_point
    
    # Initialize storage for waypoints
    waypoints = np.zeros((num_waypoints, 2))
    waypoints[0] = start_point
    
    # Try different strategies based on progress
    use_mpc = True  # Start with MPC
    
    # MPC loop
    for i in range(1, num_waypoints):
        # Progress ratio (for simple heuristic path biasing)
        progress_ratio = i / (num_waypoints - 1)
        
        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(current_state[:2] - end_point)
        close_to_goal = dist_to_goal < 0.1
        
        # If we're close to the end, switch to direct routing
        if i >= num_waypoints - 10 or close_to_goal:
            use_mpc = False
            
        if use_mpc:
            # Use MPC-based planning
            
            # Adjust weights based on progress
            current_Q_goal = Q_goal * (1 + 4 * progress_ratio)  # Increase goal weight as we progress
            current_Q_obstacle = Q_obstacle * (1.5 - 0.5 * progress_ratio)  # Decrease obstacle weight slightly
            
            # Simple reference trajectory (straight line from start to end)
            reference = start_point + progress_ratio * (end_point - start_point)
            
            # MPC variables
            x = cp.Variable((horizon + 1, state_dim))
            u = cp.Variable((horizon, control_dim))
            
            # Initialize objective
            objective = 0
            
            # Add initial state constraint
            constraints = [x[0] == current_state]
            
            # Add dynamics constraints for the prediction horizon
            for t in range(horizon):
                # Dynamics: x_{t+1} = A*x_t + B*u_t
                constraints += [x[t+1] == A @ x[t] + B @ u[t]]
                
                # Input constraints (max acceleration)
                constraints += [cp.norm(u[t], 'inf') <= max_acceleration]
                
                # Velocity constraints
                constraints += [cp.norm(x[t, 2:4], 'inf') <= max_velocity]
                
                # Goal attraction - stronger as we progress through the trajectory
                objective += current_Q_goal * cp.sum_squares(x[t, :2] - end_point)
                
                # Control input regularization
                objective += R_input * cp.sum_squares(u[t])
                
                # Obstacle avoidance using a strictly DCP-compliant approach
                for obstacle_pos, obstacle_radius in obstacles:
                    # Apply safety buffer
                    safe_radius = obstacle_radius + obstacle_buffer
                    
                    # Instead of using proximity directly, we'll use auxiliary variables
                    # For each obstacle, we penalize positions that are too close
                    
                    # Introduce slack variables
                    slack_x = cp.Variable(1)
                    slack_y = cp.Variable(1)
                    
                    # Add constraints for the slack variables
                    constraints += [
                        slack_x == x[t, 0] - obstacle_pos[0], 
                        slack_y == x[t, 1] - obstacle_pos[1]
                    ]
                    
                    # Squared distance to obstacle (always convex)
                    squared_dist = cp.sum_squares(cp.hstack([slack_x, slack_y]))
                    
                    # Penalize being too close to the obstacle (inverse reward for distance)
                    # This is convex because we're using reciprocal of a positive value
                    min_squared_dist = safe_radius**2  # Minimum acceptable squared distance
                    
                    # We'll use a smooth approximation to avoid numerical issues
                    # Instead of 1/distance, use a hyperbolic function
                    
                    # Option 1: Exponential decay penalty (strongly convex)
                    # Higher value means closer to obstacle
                    proximity_penalty = cp.exp(min_squared_dist - squared_dist)
                    
                    # Scale the penalty
                    obstacle_cost = current_Q_obstacle * proximity_penalty
                    objective += obstacle_cost
            
            # Terminal cost: strongly encourage ending close to the goal
            terminal_weight = current_Q_goal * 5
            objective += terminal_weight * cp.sum_squares(x[horizon, :2] - end_point)
            
            # Also penalize terminal velocity (to ensure smooth stopping at goal)
            objective += 5.0 * cp.sum_squares(x[horizon, 2:4])
            
            # Create the optimization problem
            problem = cp.Problem(cp.Minimize(objective), constraints)
            
            # Try to solve the problem
            try:
                if Debug:
                    print(f"Solving MPC for step {i}...")
                problem.solve(solver='ECOS')
                
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    # Get the first control input from the solution
                    control_input = u.value[0]
                    
                    # Update state using the MPC dynamics
                    current_state = A @ current_state + B @ control_input
                    
                    if Debug:
                        print(f"Step {i}: Solved with ECOS, status: {problem.status}")
                else:
                    # If solver fails, switch to direct routing
                    use_mpc = False
                    if Debug:
                        print(f"MPC solution failed at step {i}, switching to direct routing")
            except Exception as e:
                use_mpc = False
                if Debug:
                    print(f"MPC solver error at step {i}: {e}")
                    print("Switching to direct routing strategy")
        
        # If MPC failed or we're in direct routing mode
        if not use_mpc:
            if Debug:
                print(f"Using direct routing for step {i}")
            
            # Direct routing toward goal with obstacle avoidance
            direction = end_point - current_state[:2]
            direction_norm = np.linalg.norm(direction)
            
            if direction_norm > 1e-6:
                direction = direction / direction_norm
                
                # Compute repulsion from obstacles
                repulsion = np.zeros(2)
                for obstacle_pos, obstacle_radius in obstacles:
                    vec_to_obstacle = current_state[:2] - obstacle_pos
                    dist_to_obstacle = np.linalg.norm(vec_to_obstacle)
                    safe_radius = obstacle_radius + obstacle_buffer
                    
                    # Only apply repulsion if relatively close
                    if dist_to_obstacle < safe_radius * 2.0:
                        # Normalized repulsion vector
                        repulsion_vec = vec_to_obstacle / max(dist_to_obstacle, 1e-6)
                        
                        # Repulsion strength increases as we get closer
                        # Uses inverse square law
                        repulsion_strength = (safe_radius * 2.0 / max(dist_to_obstacle, safe_radius * 0.5))**2 - 1.0
                        repulsion_strength = max(0.0, repulsion_strength)
                        
                        repulsion += repulsion_vec * repulsion_strength
                
                # Combine goal attraction with obstacle repulsion
                if np.linalg.norm(repulsion) > 0:
                    repulsion = repulsion / np.linalg.norm(repulsion)
                    
                    # Bias more toward goal as we get closer to the end
                    goal_weight = 0.5 + 0.4 * progress_ratio
                    repulsion_weight = 1.0 - goal_weight
                    
                    # Near goal, mostly ignore obstacles unless very close
                    if close_to_goal:
                        goal_weight = 0.9
                        repulsion_weight = 0.1
                    
                    combined_direction = goal_weight * direction + repulsion_weight * repulsion
                    combined_direction = combined_direction / np.linalg.norm(combined_direction)
                else:
                    combined_direction = direction
                
                # Step size calculation
                if close_to_goal:
                    # Take smaller steps when close to goal
                    step_size = min(dist_to_goal * 0.5, total_distance / num_waypoints * 0.5)
                else:
                    step_size = min(dist_to_goal * 0.2, total_distance / num_waypoints)
                
                # Update position and velocity
                current_state[:2] += combined_direction * step_size
                
                # Slow down as we approach the goal
                vel_factor = min(1.0, dist_to_goal / (total_distance * 0.2))
                current_state[2:] = combined_direction * max_velocity * vel_factor
        
        # Store the new position as a waypoint
        waypoints[i] = current_state[:2]
        
        # Check if we're close enough to the goal
        dist_to_goal = np.linalg.norm(current_state[:2] - end_point)
        if dist_to_goal < goal_tolerance:
            if Debug:
                print(f"Goal reached at step {i} with distance {dist_to_goal:.4f}")
            
            # Fill remaining waypoints with goal position
            waypoints[i:] = end_point
            break
        
        # Debug information
        if Debug:
            if i % 10 == 0 or i == 1 or i == num_waypoints - 1:
                print(f"Waypoint {i}/{num_waypoints}: {waypoints[i]}, dist to goal: {dist_to_goal:.4f}")
                
                # Check distance to obstacles
                for j, (obstacle_pos, obstacle_radius) in enumerate(obstacles):
                    dist = np.linalg.norm(waypoints[i] - obstacle_pos)
                    print(f"  Distance to obstacle {j}: {dist:.3f} (min safe: {obstacle_radius + obstacle_buffer:.3f})")
    
    # Ensure the last waypoint is exactly at the goal
    waypoints[-1] = end_point
    
    return waypoints


def visualize_trajectory(
    waypoints: np.ndarray,
    obstacles: List[Tuple[np.ndarray, float]],
    start_point: np.ndarray,
    end_point: np.ndarray,
    obstacle_buffer: float = 0.2,
    title: str = "MPC Trajectory with Obstacle Avoidance"
) -> None:
    """
    Visualize the generated trajectory and obstacles.
    
    Args:
        waypoints: Array of waypoints with shape (n, 2)
        obstacles: List of obstacles specified as [(position, radius)]
        start_point: Starting point [x, y]
        end_point: Goal point [x, y]
        obstacle_buffer: Safety buffer used around obstacles
        title: Plot title
    """
    plt.figure(figsize=(10, 8))
    
    # Plot obstacles with both actual size and safety boundary
    for obstacle_pos, obstacle_radius in obstacles:
        # Actual obstacle
        circle = plt.Circle(obstacle_pos, obstacle_radius, color='r', alpha=0.7, label='Obstacle')
        plt.gca().add_patch(circle)
        
        # Safety boundary
        safety_circle = plt.Circle(obstacle_pos, obstacle_radius + obstacle_buffer, 
                                  color='r', alpha=0.2, linestyle='--', fill=True,
                                  label='Safety Boundary')
        plt.gca().add_patch(safety_circle)
    
    # Plot waypoints and trajectory
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b-', label='MPC Trajectory')
    plt.plot(waypoints[:, 0], waypoints[:, 1], 'b.', markersize=2)
    
    # Mark every 5th waypoint
    for i in range(0, len(waypoints), 5):
        plt.plot(waypoints[i, 0], waypoints[i, 1], 'bx', markersize=5)
    
    # Plot start and end points
    plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')
    plt.plot(end_point[0], end_point[1], 'mo', markersize=10, label='Goal')
    
    # Add grid and labels
    plt.grid(True)
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title(title)
    plt.axis('equal')
    
    # Create custom legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()


def save_waypoints_to_csv(
    waypoints: np.ndarray,
    filepath: str,
    z_value: float = 1.25,
    description: str = "MPC generated waypoint"
) -> None:
    """
    Save the generated waypoints to a CSV file compatible with the trajectory_following.py script.
    
    Args:
        waypoints: Array of waypoints with shape (n, 2)
        filepath: Path to save the CSV file
        z_value: Constant z-value to use for all waypoints
        description: Base description for the waypoints
    """
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['x', 'y', 'z', 'description'])
        
        for i, waypoint in enumerate(waypoints):
            writer.writerow([
                waypoint[0],
                waypoint[1],
                z_value,
                f"{description} {i+1}/{len(waypoints)}"
            ])
    
    print(f"Saved {len(waypoints)} waypoints to {filepath}")


# Example usage
if __name__ == "__main__":
    import csv
    
    # Define start and end points
    start = np.array([0.5, 0.0])
    end = np.array([0.7, 0.3])
    
    # Define obstacles as [(position, radius)]
    obstacles = [
        (np.array([0.6, 0.15]), 0.05),
        (np.array([0.65, 0.1]), 0.07),
    ]
    
    # Generate waypoints
    waypoints = generate_waypoints_mpc(
        start_point=start,
        end_point=end,
        obstacles=obstacles,
        num_waypoints=100,
        horizon=15
    )
    
    # Visualize the trajectory
    visualize_trajectory(waypoints, obstacles, start, end)
    
    # Save waypoints to CSV for robot execution
    save_waypoints_to_csv(waypoints, "mpc_trajectory.csv")
