"""
mpc_follow.py
---------------------------------
3D End-Effector MPC Tracking using CasADi.
• Robot: Panda (robosuite)
• Reference Trajectory: CSV (x,y,z,description)
• Dynamics: Simplified first-order kinematics: x_{k+1} = x_k + v_k⋅dt
• Constraints: velocity magnitude ≤ 0.15 m/s, avoid circular obstacle (radius=0.08) @ (0,0,1.25)
"""

import csv, os, time
import numpy as np
import casadi as ca
import robosuite as suite
import json

os.environ['MUJOCO_GL'] = 'glfw'

def load_default_panda_cfg():
    """Load default Panda controller configuration."""
    cfg_path = os.path.join(
        os.path.dirname(suite.__file__),
        "controllers", "config", "robots", "default_panda.json",
    )
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    return {"right": cfg}


# --------------------------------------------------------------------- #
# Load CSV trajectory (with header)
def load_traj(csv_path):
    pts, desc = [], []
    with open(csv_path, newline="") as f:
        r = csv.reader(f); next(r)      # Skip header
        for row in r:
            pts.append([float(row[0]), float(row[1]), float(row[2])])
            desc.append(row[3] if len(row) > 3 else "")
    return np.asarray(pts), desc


# --------------------------------------------------------------------- #
def make_env():
    """Create robosuite Lift environment with Panda."""
    ctrl_cfg = load_default_panda_cfg()
    ctrl_cfg["right"]["control_delta"] = True  # Enable delta control
    env = suite.make(
        "Lift",
        robots="Panda",
        controller_configs=ctrl_cfg,
        env_configuration="single-arm-opposed",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        horizon=5000,
    )
    return env


# --------------------------------------------------------------------- #
# MPC Parameters
H, dt = 15, 1/30           # Prediction horizon, simulation time step
v_max = 0.15               # Velocity constraint (m/s)
obs_center = np.array([0.0, 0.0, 1.25])  # Obstacle center
obs_radius = 0.08          # Obstacle radius

# Build CasADi MPC problem
nx, nu = 3, 3
x = ca.SX.sym("x", nx)
u = ca.SX.sym("u", nu)
f = ca.Function("f", [x, u], [x + u * dt])  # Discrete dynamics

U = ca.SX.sym("U", nu, H)
X = ca.SX.sym("X", nx, H + 1)
P = ca.SX.sym("P", nx + nx * H)      # p[0:3] = x0, the rest = references

obj, g = 0, []
Q = np.diag([40, 40, 80])            # Weight for position error
R = np.eye(nu) * 1e-2                # Weight for control effort

# Initial state constraint
g += [X[:, 0] - P[0:3]]

for k in range(H):
    x_ref = P[3 + k * 3 : 3 + (k + 1) * 3]
    # Objective function: tracking error + control effort
    obj += ca.mtimes([(X[:, k] - x_ref).T, Q, (X[:, k] - x_ref)]) + ca.mtimes([U[:, k].T, R, U[:, k]])
    # System dynamics
    g += [X[:, k + 1] - f(X[:, k], U[:, k])]
    # Velocity constraint |u| ≤ v_max
    g += [U[:, k] - v_max, -U[:, k] - v_max]
    # Obstacle avoidance: (r^2 - ||x-c||^2) ≤ 0
    g += [obs_radius ** 2 - ca.sumsqr(X[:, k] - obs_center)]

# Assemble NLP
OPT = {"ipopt.print_level": 0, "ipopt.sb": "yes"}
nlp = {"f": obj, "x": ca.vertcat(U.reshape((-1, 1)), X.reshape((-1, 1))), "p": P, "g": ca.vertcat(*g)}
solver = ca.nlpsol("solver", "ipopt", nlp, OPT)

# Bounds for constraints
lbg = np.zeros((len(g), 1))
ubg = np.zeros((len(g), 1))
# Velocity bounds are enforced via g
lbx = -np.inf * np.ones(U.numel() + X.numel())
ubx = np.inf * np.ones(U.numel() + X.numel())


# --------------------------------------------------------------------- #
def solve_mpc(x0, ref):
    """Solve MPC for a given current state and future reference trajectory.
    Args:
        x0: current position, shape (3,)
        ref: future reference points, shape (H, 3)
    Returns:
        Optimal velocity command, shape (3,)
    """
    p = np.concatenate([x0, ref.flatten()])
    sol = solver(x0=np.zeros(lbx.shape), p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    u_opt = np.array(sol["x"][0:nu]).flatten()
    return u_opt


# --------------------------------------------------------------------- #
def main(csv_path):
    traj, desc = load_traj(csv_path)
    env = make_env()
    obs = env.reset()
    idx = 0

    while idx < len(traj):
        # Get current end-effector position
        cur = env._eef_xpos.copy()
        # Construct future reference window
        future = np.vstack([traj[min(idx + k, len(traj) - 1)] for k in range(H)])
        # Solve for MPC action
        vel = solve_mpc(cur, future)
        # Form full 7D action (delta position + fixed orientation + gripper)
        action = np.concatenate([vel, obs["robot0_eef_quat"], [0]])
        obs, reward, done, _ = env.step(action)
        env.render()

        # Check if the current reference point is reached
        if np.linalg.norm(cur - traj[idx]) < 0.02:
            idx += 1
            print(f"Reached waypoint {idx}/{len(traj)}:", desc[idx - 1])
        if done:
            break

    env.close()


if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "complex_trajectory.csv")
    main(csv_file)
