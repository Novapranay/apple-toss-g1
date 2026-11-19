# Unitree G1 Humanoid Tossing an Apple (MuJoCo)

A MuJoCo simulation in which a **Unitree G1 humanoid** tosses an apple from one hand to the other, alternating hands once per second, and renders the result to an MP4 video.

**Team:** Archit Deshmukh, Jainil Patel, Pranay Singh Thakur

## How it works

| Piece | Approach |
|---|---|
| Robot | Official Unitree G1 (29-DoF) model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1) (BSD-3-Clause), downloaded automatically |
| Model edits | The script writes a modified copy of the G1 XML that adds an apple body and two sites per hand (palm-centre "apple" site and an "up" site used to keep the palm facing upward). The Menagerie keyframe is removed so it matches the added apple |
| Pose | The robot stands in place; legs and waist are fixed at the zero pose (kinematic, not balance-controlled) |
| Arms | Both 7-DoF arms (3 shoulder, elbow, 3 wrist) follow a periodic cubic-spline hand path. Damped-least-squares inverse kinematics solves each step so the apple site reaches the path and the palm faces up. Arm joint angles are set directly (kinematic) |
| Apple | A real MuJoCo free body. While held it rides on the palm. At release it gets the exact ballistic velocity that lands it on the other palm 0.5 s later (Euler integrator, discrete-time correction), then MuJoCo integrates the flight under gravity |
| Output | `apple_toss_g1.mp4`, 1280x720, 50 fps, 12 s (about 11 tosses) |

The script prints the worst-case IK position error and palm tilt so you can check how well the hands track their targets.

## Requirements

- Python 3.9+ (tested target: Google Colab)
- `git` and internet access (to fetch MuJoCo Menagerie)
- Python packages in `requirements.txt`: `mujoco>=3.2`, `numpy`, `scipy`, `imageio`, `imageio-ffmpeg`
- Headless rendering: EGL (GPU) or OSMesa on Linux; a normal OpenGL desktop works on Windows/macOS

## Run on Google Colab (recommended)

1. Open <https://colab.research.google.com> and choose **File -> Upload notebook**.
2. Upload `apple_toss_g1_colab.ipynb` from this repo.
3. (Optional) **Runtime -> Change runtime type -> T4 GPU** for faster rendering.
4. **Runtime -> Run all**. The first cell installs packages, the second runs the simulation and shows `apple_toss_g1.mp4` inline.
5. Download the video from the file browser (left sidebar) or with:
   ```python
   from google.colab import files; files.download('apple_toss_g1.mp4')
   ```

If rendering fails with an EGL error, run `!apt-get -qq install libosmesa6-dev` in a cell above, and change `"egl"` to `"osmesa"` on line 2 of the script.

## Run locally

```bash
git clone <this-repo-url>
cd apple-toss-g1
python -m venv venv
# Windows: venv\Scripts\activate      Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python apple_toss_g1.py
```

The script clones MuJoCo Menagerie into `./mujoco_menagerie` on first run, then writes `apple_toss_g1.mp4`.

## Repository layout

```
apple_toss_g1.py              # full simulation: model setup, IK, toss logic, rendering
apple_toss_g1_colab.ipynb     # Colab notebook wrapping the same code
requirements.txt
results/                      # put the rendered video here
```

## Tunable parameters (top of `apple_toss_g1.py`)

- `DURATION`, `FPS`, `TOSS_PERIOD`, `FLIGHT_STEPS` - timing of the video and tosses
- `P_PTS` - waypoints of the hand path (metres, world frame; the G1 stands at the origin facing +x)
- camera: `cam.azimuth`, `cam.distance`, `cam.elevation` in `main()`

## Known limitations

- The apple rests on the palm and is released/caught by script; there is no finger grasping or contact-based catch.
- The robot is not balance-controlled; only the arms move.
- The hand path coordinates were chosen by estimate for the G1's arm length and may need adjustment if the printed IK error or palm tilt is large.

## Credits

Robot model: Unitree G1 from Google DeepMind's MuJoCo Menagerie (BSD-3-Clause). Physics: [MuJoCo](https://mujoco.org).

