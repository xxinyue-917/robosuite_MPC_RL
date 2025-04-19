"""
MPC End-Effector Tracking for Panda Robot
- Robosuite 1.5.1
- BASIC Controller
- CasADi + IPOPT
"""

import os, time, csv
import numpy as np
import casadi as ca
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

# ---------------------------------------------------------------
# 1. Create Environment
def make_env():
    ctrl_cfg = load_composite_controller_config(controller="BASIC")
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        controller_configs=ctrl_cfg,
        env_configuration="default",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=False,
        horizon=5000,
    )
    return env

# ---------------------------------------------------------------
# 2. Load Trajectory
def load_traj(csv_path):
    pts, desc = [], []
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            pts.append([float(x) for x in row[:3]])
            desc.append(row[3] if len(row) > 3 else "")
    return np.array(pts), desc

# ---------------------------------------------------------------
# 3. Build MPC Model
H = 15                      # Horizon
dt = 1/30                   # Timestep
v_max = 0.15                # Max velocity (m/s)

nx = nu = 3
x = ca.SX.sym("x", nx)
u = ca.SX.sym("u", nu)
f = ca.Function("f", [x, u], [x + u * dt])

U = ca.SX.sym("U", nu, H)
X = ca.SX.sym("X", nx, H+1)
P = ca.SX.sym("P", nx + nx*H)

Q = np.diag([40, 40, 80])
R = np.eye(3) * 1e-2

obj = 0
g = []
g.append(X[:, 0] - P[0:3])  # initial constraint

for k in range(H):
    ref = P[3+3*k : 3+3*(k+1)]
    obj += ca.mtimes([(X[:,k]-ref).T, Q, (X[:,k]-ref)]) + ca.mtimes([U[:,k].T, R, U[:,k]])
    g.append(X[:,k+1] - f(X[:,k], U[:,k]))

nlp = {"f": obj,
       "x": ca.vertcat(U.reshape((-1,1)), X.reshape((-1,1))),
       "p": P,
       "g": ca.vertcat(*g)}

solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt.print_level": 0})

# Variable bounds
lbx = []
ubx = []
for _ in range(H):
    lbx += [-v_max] * 3   # for U
    ubx += [ v_max] * 3
for _ in range(H+1):
    lbx += [-np.inf] * 3  # for X
    ubx += [ np.inf] * 3
lbx = np.array(lbx).reshape((-1,1))
ubx = np.array(ubx).reshape((-1,1))

# Constraint bounds
ng = 3 * (H+1)  # 3D for each state constraint
lbg = np.zeros((ng, 1))
ubg = np.zeros((ng, 1))

# ---------------------------------------------------------------
# 4. Solve MPC
def solve_mpc(x0, ref_window):
    p_vec = np.concatenate([x0, ref_window.flatten()])
    sol = solver(x0=np.zeros(lbx.shape), p=p_vec,
                 lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    return np.array(sol["x"][:nu]).flatten()

# ---------------------------------------------------------------
# 5. Main Function
def main(csv_path):
    traj, desc = load_traj(csv_path)
    env = make_env()
    obs = env.reset()

    # Optional teleport to start
    mocap_body = "right_eef_target"
    start_pos = traj[0]
    env.sim.data.set_mocap_pos(mocap_body, start_pos)
    env.sim.forward()
    time.sleep(0.5)

    idx = 0
    threshold = 0.02  # 2 cm
    print(f"Trajectory loaded with {len(traj)} points.\n")

    step_count = 0
    max_steps = 4000

    while idx < len(traj) and step_count < max_steps:
        cur = obs["robot0_eef_pos"].copy()

        # Build future window
        future = np.vstack([traj[min(idx+k, len(traj)-1)] for k in range(H)])

        vel_cmd = solve_mpc(cur, future)

        action = np.concatenate([vel_cmd, np.zeros(3), [0.0]])
        obs, reward, done, _ = env.step(action)
        env.render()

        dist = np.linalg.norm(cur - traj[idx])

        if dist < threshold:
            print(f"✅ Reached point {idx}: {desc[idx]} (dist={dist:.3f})")
            idx += 1

        if done:
            print("Environment done signal received!")
            break

        step_count += 1
        time.sleep(0.01)

    env.close()
    print("\nMPC Trajectory Following Completed!")

# ---------------------------------------------------------------
if __name__ == "__main__":
    csv_file = os.path.join(os.path.dirname(__file__), "complex_trajectory.csv")
    main(csv_file)
