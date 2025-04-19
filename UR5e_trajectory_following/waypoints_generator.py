import numpy as np
import cvxpy as cp
import dccp
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
# Save waypoints to CSV file
import csv
import os

def generate_mpc_waypoints_dccp(
    start: np.ndarray,
    end: np.ndarray,
    obstacle_positions: list[np.ndarray],
    obstacle_radii: list[float],
    horizon: int = 20,
    dt: float = 1.0,
    w_goal: float = 1.0,
    w_control: float = 0.1,
    max_iter: int = 50,
    eps: float = 1e-3
) -> np.ndarray:
    # decision variables
    x = cp.Variable((2, horizon+1))
    u = cp.Variable((2, horizon))

    cost = 0
    constraints = []
    # boundary conditions
    constraints += [x[:,0] == start]
    constraints += [x[:,horizon] == end]

    # dynamics + cost
    for k in range(horizon):
        constraints += [x[:,k+1] == x[:,k] + dt*u[:,k]]
        cost += w_control*cp.sum_squares(u[:,k])
    cost += w_goal*cp.sum_squares(x[:,horizon] - end)

    # nonconvex obstacle‐avoidance
    for k in range(horizon+1):
        for obs, r in zip(obstacle_positions, obstacle_radii):
            constraints += [cp.norm(x[:,k] - obs) >= r]

    prob = cp.Problem(cp.Minimize(cost), constraints)

    # solve via DCCP, using SCS internally so we don't pass DCCP flags to Clarabel
    result = prob.solve(
        method='dccp',
        solver=cp.SCS,       # <— pick a solver that won’t choke on our keywords
        max_iter=max_iter,   # DCCP’s iteration cap
        eps=eps,             # DCCP’s feasibility tolerance
        verbose=True
    )

    if prob.status not in ["Solved", "Converged"]:
        raise RuntimeError(f"DCCP failed: status={prob.status}")

    return x.value

if __name__ == "__main__":
    # define problem
    start = np.array([0.0, 0.0])
    end   = np.array([10.0, 10.0])
    obs_positions = [np.array([5.0, 5.0]), np.array([7.0, 3.0])]
    obs_radii     = [1.0, 1.0]

    # generate
    waypoints = generate_mpc_waypoints_dccp(
        start, end, obs_positions, obs_radii,
        horizon=30, dt=0.5,
        w_goal=10.0, w_control=0.1,
        max_iter=100, eps=1e-4
    )

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "mpc_trajectory.csv")

    # Write waypoints to CSV file
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write header
        csv_writer.writerow(['x', 'y', 'z', 'description'])
        
        # Write waypoints
        for i in range(waypoints.shape[1]):
            x, y = waypoints[:, i]
            # Add z coordinate (1.25) and description
            csv_writer.writerow([x, y, 1.25, f"MPC generated waypoint {i+1}/{waypoints.shape[1]}"])
    
    print(f"Saved {waypoints.shape[1]} waypoints to {csv_path}")

    # plot
    fig, ax = plt.subplots(figsize=(6,6))

    # trajectory
    ax.plot(waypoints[0,:], waypoints[1,:], '-o', label='Trajectory')

    # start & end
    ax.scatter(*start, c='green', s=100, label='Start')
    ax.scatter(*end,   c='red',   s=100, label='End')

    # obstacles
    for (ox, oy), r in zip(obs_positions, obs_radii):
        circle = Circle((ox, oy), r, color='gray', alpha=0.5)
        ax.add_patch(circle)
        # optional: draw obstacle boundary
        ax.add_patch(Circle((ox, oy), r, fill=False, edgecolor='k'))

    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('MPC‐Generated Waypoints with Obstacles')
    ax.legend()
    ax.grid(True)
    plt.show()
