import os, re, subprocess
os.environ.setdefault("MUJOCO_GL", "egl")   # headless rendering on Colab; use "osmesa" if EGL errors

import numpy as np
import mujoco
import imageio
from scipy.interpolate import CubicSpline

# ------------------------------------------------------------ 1. get the Unitree G1 (MuJoCo Menagerie, BSD-3)
if not os.path.isdir("mujoco_menagerie/unitree_g1"):
    subprocess.run("git clone -q --depth 1 --filter=blob:none --sparse "
                   "https://github.com/google-deepmind/mujoco_menagerie.git", shell=True, check=True)
    subprocess.run("cd mujoco_menagerie && git sparse-checkout set unitree_g1", shell=True, check=True)
G1 = os.path.abspath("mujoco_menagerie/unitree_g1")

W, H = 1280, 720
g1 = open(f"{G1}/g1.xml").read()
scene = open(f"{G1}/scene.xml").read()

g1 = g1.replace('<option integrator="implicitfast"/>', f'<option integrator="Euler" timestep="{0.005}"/>')
g1 = re.sub(r"<keyframe>.*?</keyframe>", "", g1, flags=re.S)      # keyframe size would not match the added apple

# palm-centre "apple" sites (apple rests on the palm side of the hand) + an "up" site 10 cm above it
for side, y in (("left", 0.003), ("right", -0.003)):
    s = 1 if side == "left" else -1          # palm faces -y (left hand) / +y (right hand) in wrist frame
    anchor = f'<geom pos="0.0415 {y} 0" quat="1 0 0 0" class="visual" mesh="{side}_rubber_hand"/>'
    assert anchor in g1, anchor
    g1 = g1.replace(anchor, anchor + f"""
                            <site name="apple_{side}" pos="0.085 {-0.055 * s} 0" size="0.008" rgba="0 1 0 0"/>
                            <site name="up_{side}"    pos="0.085 {-0.155 * s} 0" size="0.008" rgba="0 1 0 0"/>""")

apple = """
      <body name="apple" pos="0.3 0 1.2">
        <freejoint name="apple_free"/>
        <geom type="sphere" size="0.035" mass="0.15" rgba=".85 .06 .06 1" contype="0" conaffinity="0"/>
        <geom type="capsule" fromto="0 0 0.03 0.004 0 0.052" size="0.004" rgba=".35 .2 .1 1" mass="0" contype="0" conaffinity="0"/>
      </body>
    </worldbody>"""
g1 = g1.replace("</worldbody>", apple, 1)
open(f"{G1}/g1_toss.xml", "w").write(g1)

scene = scene.replace('<include file="g1.xml"/>', '<include file="g1_toss.xml"/>')
scene = scene.replace('<global azimuth="140" elevation="-20"/>',
                      f'<global azimuth="140" elevation="-20" offwidth="{W}" offheight="{H}"/>')
open(f"{G1}/scene_toss.xml", "w").write(scene)

# ------------------------------------------------------------ 2. parameters
DT = 0.005                 # physics step (s)
DURATION = 12.0
FPS = 50                   # render every 4 steps
FLIGHT_STEPS = 100         # 0.5 s flight
TOSS_PERIOD = 1.0          # one toss per second, alternating hands
FIRST_TOSS = 0.4
OUT = "apple_toss_g1.mp4"
UP = np.array([0.0, 0.0, 1.0])

# LEFT-hand apple path (u in [0,2)): throw at u=0, catch at u=1.5. Right hand = mirror (y -> -y), shifted 1 s.
U_PTS = np.array([0.0, 0.5, 1.0, 1.25, 1.5, 2.0])
P_PTS = np.array([
    [0.26, 0.17, 0.82],    # 0.0 throw
    [0.28, 0.14, 0.92],    # 0.5 follow-through
    [0.27, 0.17, 0.90],    # 1.0 ready
    [0.30, 0.10, 0.98],    # 1.25 reach
    [0.30, 0.07, 0.92],    # 1.5 catch (moving down = cushion)
    [0.26, 0.17, 0.82],    # 2.0 = throw again
])
SPLINE = CubicSpline(U_PTS, P_PTS, bc_type="periodic")


def hand_target(side, t):
    if side == "left":
        return SPLINE((t - FIRST_TOSS) % 2.0)
    p = SPLINE((t - FIRST_TOSS - TOSS_PERIOD) % 2.0).copy()
    p[1] = -p[1]
    return p


# ------------------------------------------------------------ 3. IK: apple site -> target, palm normal -> up
class ArmIK:
    JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]

    def __init__(self, model, side):
        sg = 1 if side == "left" else -1
        names = [f"{side}_{j}_joint" for j in self.JOINTS]
        ids = [model.joint(n).id for n in names]
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.dof = np.array([model.jnt_dofadr[i] for i in ids])
        self.lo, self.hi = model.jnt_range[ids, 0], model.jnt_range[ids, 1]
        self.sa = model.site(f"apple_{side}").id
        self.sb = model.site(f"up_{side}").id
        self.rest = np.array([-0.3, 0.15 * sg, 0.0, 0.3, -1.4 * sg, 0.0, 0.0])
        self.q = self.rest.copy()

    def solve(self, model, data, target, lam=0.05, w=0.7, iters=80):
        jA = np.zeros((3, model.nv)); jB = np.zeros((3, model.nv))
        q = self.q.copy()
        for _ in range(iters):
            data.qpos[self.qadr] = q
            mujoco.mj_forward(model, data)
            pA, pB = data.site_xpos[self.sa], data.site_xpos[self.sb]
            e_pos = target - pA
            e_or = 0.1 * UP - (pB - pA)
            if np.linalg.norm(e_pos) < 5e-4 and np.linalg.norm(e_or) < 5e-3:
                break
            mujoco.mj_jacSite(model, data, jA, None, self.sa)
            mujoco.mj_jacSite(model, data, jB, None, self.sb)
            J = np.vstack([jA[:, self.dof], w * (jB - jA)[:, self.dof]])
            err = np.concatenate([e_pos, w * e_or])
            Jp = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(6))
            dq = Jp @ err + (np.eye(7) - Jp @ J) @ (0.1 * (self.rest - q))
            q = np.clip(q + 0.8 * dq, self.lo, self.hi)
        data.qpos[self.qadr] = q
        mujoco.mj_forward(model, data)
        self.q = q
        pA, pB = data.site_xpos[self.sa].copy(), data.site_xpos[self.sb]
        tilt = np.degrees(np.arccos(np.clip((pB - pA)[2] / 0.1, -1, 1)))
        return q.copy(), pA, tilt


# ------------------------------------------------------------ 4. simulate + render
def main():
    model = mujoco.MjModel.from_xml_path(f"{G1}/scene_toss.xml")
    data = mujoco.MjData(model)
    n_steps = int(round(DURATION / DT))
    sides = ("left", "right")
    iks = {s: ArmIK(model, s) for s in sides}

    qa = model.jnt_qposadr[model.joint("apple_free").id]
    va = model.jnt_dofadr[model.joint("apple_free").id]
    base_q = model.qpos0[:qa].copy()          # G1 standing pose (zero joint angles), apple excluded

    # stage 1: arm angles + apple-site (palm) positions for every step
    q_traj = {s: np.zeros((n_steps, 7)) for s in sides}
    site = {s: np.zeros((n_steps, 3)) for s in sides}
    worst_pos = worst_tilt = 0.0
    for n in range(n_steps):
        t = n * DT
        for s in sides:
            tgt = hand_target(s, t)
            q_traj[s][n], site[s][n], tilt = iks[s].solve(model, data, tgt)
            worst_pos = max(worst_pos, np.linalg.norm(site[s][n] - tgt))
            worst_tilt = max(worst_tilt, tilt)
    print(f"IK worst position error {worst_pos * 1000:.1f} mm, worst palm tilt {worst_tilt:.1f} deg")

    # stage 2: toss schedule with exact discrete-time ballistic velocity
    g = -model.opt.gravity[2]
    N = FLIGHT_STEPS
    events, k = {}, 0
    while True:
        n_t = int(round((FIRST_TOSS + k * TOSS_PERIOD) / DT))
        n_c = n_t + N
        if n_c >= n_steps:
            break
        thrower, catcher = ("left", "right") if k % 2 == 0 else ("right", "left")
        events[n_t] = ("throw", thrower, catcher, n_c)
        events[n_c] = ("catch", catcher)
        k += 1

    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.2, 0.0, 0.85]
    cam.distance = 2.2
    cam.azimuth = 150
    cam.elevation = -10

    # Euler integrator + no forces but gravity on the apple -> v0 above lands it exactly on the catching palm.
    holder = "left"
    every = int(round(1.0 / (FPS * DT)))
    frames = []
    for n in range(n_steps):
        data.qpos[:qa] = base_q                 # robot is posed kinematically (stands in place)
        data.qvel[:va] = 0.0
        for s in sides:
            data.qpos[iks[s].qadr] = q_traj[s][n]

        ev = events.get(n)
        if ev and ev[0] == "throw":
            _, thrower, catcher, n_c = ev
            p0, p1 = site[thrower][n], site[catcher][n_c]
            grav_disp = np.array([0, 0, -g]) * DT**2 * N * (N + 1) / 2   # discrete Euler stepping
            v0 = (p1 - p0 - grav_disp) / (N * DT)
            data.qpos[qa:qa + 3] = p0
            data.qvel[va:va + 3] = v0
            holder = None
        elif ev and ev[0] == "catch":
            holder = ev[1]

        if holder is not None:
            data.qpos[qa:qa + 3] = site[holder][n]
            data.qvel[va:va + 6] = 0.0
        data.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]

        mujoco.mj_forward(model, data)
        if n % every == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        mujoco.mj_step(model, data)

    with imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=1) as wr:
        for f in frames:
            wr.append_data(f)
    print(f"wrote {OUT}: {len(frames)} frames, {len(frames) / FPS:.1f} s")


main()

