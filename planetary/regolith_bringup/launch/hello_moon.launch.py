# Copyright 2026 Regolith Project contributors
# SPDX-License-Identifier: Apache-2.0
"""The full Regolith hello-world demo: procedural lunar terrain, a localized skid-steer rover, a traversability costmap, and autonomous navigation - either click-a-goal-yourself (the default) or a scripted multi-waypoint tour.

    ros2 launch regolith_bringup hello_moon.launch.py seed:=42
    ros2 launch regolith_bringup hello_moon.launch.py seed:=42 mission:=tour

With `mission:=tour` (default `none`), a fixed 5-waypoint loop starts
automatically after a short delay - see scripts/tour_mission.py. Otherwise,
use RViz's "2D Goal Pose" tool to send goals yourself, same as
autonomous_demo.launch.py (which this supersedes as the one-command entry
point - see scripts/demo.sh). RViz opens automatically with the rover config;
pass rviz:=false to skip it (e.g. for headless runs).
"""

import atexit
import fcntl
import os
from pathlib import Path
import random
import subprocess

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.actions import RegisterEventHandler
from launch.actions import Shutdown
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

ROVER_NAME = "rover"
WORLD_NAME = "regolith_moon"

# Registry of ROS_DOMAIN_IDs currently claimed by live regolith launches. Each
# claimed id N is a file "N.lock" here containing the claiming launch's PID; a
# ".registry.lock" flock serialises the claim so two launches racing each other
# can never pick the same id. See _allocate_domain_id below and PROGRESS.md's
# "Per-launch ROS_DOMAIN_ID isolation" section.
_DOMAIN_REGISTRY_DIR = Path(os.path.expanduser("~/.ros/regolith_domain_ids"))
# 1-101, skipping 0: 0 is the DDS default domain every un-configured ROS process
# on the box lands on, so avoiding it keeps us clear of unrelated ROS traffic
# too. 0-101 is the Linux-safe range (higher ids collide with the ephemeral port
# range); see the design note in PROGRESS.md.
_DOMAIN_ID_MIN = 1
_DOMAIN_ID_MAX = 101


def _pid_alive(pid: int) -> bool:
    """Return True if a process with this PID currently exists (signal 0 probes it)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists but owned by someone else - still "alive" for our purposes.
        return True
    return True


def _release_domain_claim(lock_path: Path) -> None:
    """Best-effort removal of our own claim file on clean interpreter exit.

    Not relied on for correctness: if the launch is SIGKILLed (or otherwise dies
    without running atexit handlers) the claim file is left behind, but the next
    launch reclaims it via the _pid_alive() staleness check, so a leaked file is
    self-healing rather than a permanent leak of a domain id.
    """
    try:
        if lock_path.read_text().strip().split()[0] == str(os.getpid()):
            lock_path.unlink()
    except (OSError, IndexError):
        pass


def _allocate_domain_id() -> int:
    """Claim a ROS_DOMAIN_ID no concurrently-live regolith launch is using and export it into os.environ, so every process this launch subsequently spawns (gz sim, the ros_gz bridge, EKF, planner, rviz, ...) shares one DDS domain that is isolated from any other launch's domain.

    This is the structural fix for the overnight-freeze root cause (two launches
    sharing one ROS graph over identical topic names - see PROGRESS.md). A
    directory of PID-tagged lock files, guarded by a single flock, makes the
    claim atomic across processes: two launches started at the same instant
    serialise on the flock and are guaranteed distinct ids. Stale claims from a
    crashed launch are reclaimed via a liveness check on the recorded PID.

    An explicitly-set ROS_DOMAIN_ID in the environment is honoured as-is (the
    user asked for that specific domain); we still record it in the registry
    best-effort so a concurrent auto-allocation avoids it, but we never refuse
    or override it.
    """
    _DOMAIN_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    registry_lock_fd = os.open(
        str(_DOMAIN_REGISTRY_DIR / ".registry.lock"), os.O_CREAT | os.O_RDWR, 0o644
    )
    try:
        fcntl.flock(registry_lock_fd, fcntl.LOCK_EX)

        preset = os.environ.get("ROS_DOMAIN_ID", "").strip()
        if preset:
            try:
                preset_id = int(preset)
            except ValueError:
                preset_id = None
            if preset_id is not None:
                # Best-effort claim; do not refuse if already held (explicit
                # user intent wins over our collision-avoidance).
                lock_path = _DOMAIN_REGISTRY_DIR / f"{preset_id}.lock"
                if not lock_path.exists():
                    lock_path.write_text(f"{os.getpid()} {preset_id}\n")
                    atexit.register(_release_domain_claim, lock_path)
                return preset_id

        candidates = list(range(_DOMAIN_ID_MIN, _DOMAIN_ID_MAX + 1))
        random.shuffle(candidates)
        for domain_id in candidates:
            lock_path = _DOMAIN_REGISTRY_DIR / f"{domain_id}.lock"
            if lock_path.exists():
                try:
                    holder_pid = int(lock_path.read_text().strip().split()[0])
                except (OSError, ValueError, IndexError):
                    holder_pid = None
                if holder_pid is not None and _pid_alive(holder_pid):
                    continue  # genuinely in use by another live launch
                # Stale claim from a crashed/killed launch - reclaim it. Safe:
                # we hold the registry flock, so no other launch is scanning.
                try:
                    lock_path.unlink()
                except OSError:
                    continue
            lock_path.write_text(f"{os.getpid()} {domain_id}\n")
            atexit.register(_release_domain_claim, lock_path)
            os.environ["ROS_DOMAIN_ID"] = str(domain_id)
            return domain_id

        # All 101 ids claimed by live launches (would need 101 concurrent
        # regolith sims). No isolation guarantee is possible here; fall back to a
        # random pick and let the collision happen rather than refusing to launch.
        domain_id = random.randint(_DOMAIN_ID_MIN, _DOMAIN_ID_MAX)
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        return domain_id
    finally:
        fcntl.flock(registry_lock_fd, fcntl.LOCK_UN)
        os.close(registry_lock_fd)


def _bake_rover_model_sdf(rover_urdf_path: Path, spawn_z: float) -> str:
    """Convert the rover URDF into an SDF <model> block, renamed to ROVER_NAME and posed at its spawn point, ready to splice directly into the generated world.sdf (the same file rocks/terrain are already baked into - see worldgen.py).

    This makes the rover part of the world from the moment gz-sim loads it, instead
    of being added ~3s later via a separate `ros2 run ros_gz_sim create` service
    call - one less moving part, and the generated world.sdf is now fully
    self-contained (no runtime spawn dependency). Note: an earlier version of this
    docstring claimed this fixes a GUI scene-sync bug where an already-open GUI
    window never renders entities added after it starts; that theory did not
    survive re-testing (the rover still wasn't visibly rendering after this change)
    and was wrong - see PROGRESS.md's "Gazebo shows nothing but terrain" section for
    the real root cause (the default camera was ~155m from spawn) and its fix (in
    worldgen.py's build_world_sdf). This function is kept because baking the rover
    in is still a reasonable simplification on its own, not because it fixes that
    bug.
    """
    converted = subprocess.run(
        ["gz", "sdf", "-p", str(rover_urdf_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # gz sdf -p wraps its output in an <sdf version='...'> root element; strip that
    # down to the bare <model>...</model>, since we're splicing this directly inside
    # world.sdf's own <world> element (nesting a second <sdf> in there is invalid -
    # gz-sim logs "not defined in SDF" and silently ignores the whole block).
    start = converted.index("<model ")
    end = converted.rindex("</model>") + len("</model>")
    model_sdf = converted[start:end]
    # xacro's <robot name="regolith_rover"> becomes the SDF model's name; rename it to
    # ROVER_NAME to match the bridge/flip-recovery topic remappings below, and give it
    # a spawn pose (gz sdf -p emits none, since a bare URDF has no world placement).
    marker = "<model name='regolith_rover'>"
    if not model_sdf.startswith(marker):
        raise RuntimeError(
            "gz sdf -p produced unexpected output - couldn't find the "
            f"'regolith_rover' model tag to rename/pose:\n{model_sdf[:500]}"
        )
    return (
        f"<model name='{ROVER_NAME}'>\n    <pose>0 0 {spawn_z} 0 0 0</pose>"
        + model_sdf[len(marker) :]
    )


CINE_RIG_MODEL_NAME = "cine_rig"


def _cine_rig_sdf(spawn_x: float, spawn_y: float, spawn_z: float, hfov: float = 1.3) -> str:
    """A free-flying, camera-only model for documentary-style footage of a run -
    chase/orbit/crane/track, driven each tick by camera_rig_node.py via gz-sim's
    `/world/<world>/set_pose` service (see that node's module docstring for the full
    mechanism writeup and how it was verified on this install).

    Deliberately NOT attached to the rover by any joint, and does not touch the
    rover's mass, inertia, collisions or sensors - the "frozen physics" constraint
    that governs everything else in this file. It is a plain <static> model so
    set_pose only ever moves a visual/camera, never anything the physics engine
    steps; confirmed on this gz-sim 8.14 install that a static model's pose CAN be
    teleported this way (it was worth checking - a plausible silent no-op).

    Initial pose here barely matters: camera_rig_node.py takes over within its
    first tick once /ground_truth/pose starts arriving. It is placed a few metres
    above the spawn point purely so nothing looks wrong in the sliver of time
    before that.

    far=6000, matching scripts/render_still.py's own cine rigs: the sky dome and
    far-field horizon extend out past 4000+ m and would clip/vanish inside a
    shorter far plane - see that script's comment on the same constant.

    update_rate=30, matching camera_rig_node.py's own default control-tick rate
    (`update_rate_hz`) - the two MUST match, not just be close. A 30 fps clip built
    from a sensor whose pose only changes at 15 Hz repeats every other frame at the
    same pose, which is not 30 fps motion no matter what the container says - see
    docs/media/README.md's "frame count, not container fps" note, where exactly
    this was measured and caught on an earlier 15 Hz take (26 distinct frames in a
    1.04 s / ~31-output-frame clip - visibly stuttery, would only get worse on a
    longer hero take). 30 Hz costs more render time per simulated second than 15
    did; that cost is measured and reported in the same README section.

    `hfov` defaults to 1.3 (chase/crane/track's working value, confirmed good framing
    on the approved hero chase clip) but the orbit shot passes a narrower value - a
    first orbit take at radius 9 m / height 4 m / hfov 1.3 put the rover at ~3% of
    frame width, technically clean footage that failed at its one job as a beauty
    shot (see docs/media/README.md). Narrowing hfov and tightening orbit_radius/
    orbit_height together (rather than either alone) gets the rover back to a
    16-30%-of-frame-width hero size while keeping enough standoff to still read as
    an orbit, not a close-up - checked with a still render before the reshoot.
    """
    return f"""    <model name="{CINE_RIG_MODEL_NAME}">
      <static>true</static>
      <pose>{spawn_x:.3f} {spawn_y - 3.0:.3f} {spawn_z + 2.0:.3f} 0 0.3 0</pose>
      <link name="link">
        <sensor name="cine_rig_cam" type="camera">
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <topic>cine_rig</topic>
          <camera>
            <horizontal_fov>{hfov:.4f}</horizontal_fov>
            <image><width>1280</width><height>720</height></image>
            <clip><near>0.05</near><far>6000</far></clip>
          </camera>
          <plugin filename="gz-sim-camera-video-recorder-system" name="gz::sim::systems::CameraVideoRecorder">
            <service>rover/rig/record_video</service>
          </plugin>
        </sensor>
      </link>
    </model>
"""


def _generate_and_launch(context, *args, **kwargs):
    import dataclasses
    import json

    from regolith_terrain_gen.cli import default_output_dir
    from regolith_terrain_gen.config import TerrainConfig
    from regolith_terrain_gen.generate import generate_world
    import xacro

    # Claim a private ROS_DOMAIN_ID *first*, before any of the actions returned
    # below are built or spawned. Mutating os.environ here (inside the running
    # OpaqueFunction) means every subsequently-spawned process - the Nodes returned
    # at the bottom of this function AND the ones inside the included
    # ros_gz_sim/gz_sim.launch.py - captures the new value at spawn time, so the
    # whole launch tree shares one DDS domain isolated from any other launch. This
    # makes the overnight-freeze failure mode (two launches merging into one ROS
    # graph over shared topic names) structurally impossible between any two
    # launches that each get a distinct id. See PROGRESS.md.
    domain_id = _allocate_domain_id()
    print(
        f"[hello_moon.launch] Using ROS_DOMAIN_ID={domain_id} "
        f"(isolated DDS domain for this launch - see PROGRESS.md)",
        flush=True,
    )

    raw_seed = LaunchConfiguration("seed").perform(context)
    try:
        seed = int(raw_seed)
        if seed < 0:
            raise ValueError
    except ValueError:
        raise RuntimeError(
            f"Launch argument 'seed' must be a non-negative integer, got '{raw_seed}'"
        ) from None
    cfg = TerrainConfig(seed=seed)
    # Media capture only: the shipped lighting is a 12-degree sun over a near-black
    # ambient, which is what a low lunar sun actually looks like and what every
    # measurement in this repo was taken under. It also renders most of the surface
    # texture as unreadable shadow on video. cine_light raises the sun and lifts the
    # ambient off the floor for footage; it is render-only (the heightmap, the rocks,
    # the collision boxes and the costmap are all generated before the light is
    # written) so a clip recorded with it is driving the same world as a normal run.
    if LaunchConfiguration("cine_light").perform(context).lower() == "true":
        # 25 deg sun / 0.12 ambient (a first pass) blew the regolith surface out to
        # near-white - real lunar regolith albedo is ~0.08-0.14, closer to worn
        # asphalt than concrete; Apollo photos read bright because of direct
        # unfiltered sun and camera exposure, not because the ground is pale. 18
        # deg / 0.07 keeps the low-sun long-shadow character and enough fill to
        # read the surface on video, without crushing shadows to flat black or
        # blowing the lit faces past a real regolith tone.
        cfg = dataclasses.replace(
            cfg, sun_elevation_deg=18.0, scene_ambient=(0.07, 0.07, 0.08)
        )
    output_dir = default_output_dir(seed)
    world_sdf_path = generate_world(cfg, output_dir, start_paused=False)
    manifest_path = output_dir / "manifest.json"

    # Spawn height must be above the ACTUAL local terrain elevation, not a fixed
    # guess - see PROGRESS.md M2 and heightmap.py's build_terrain_collision_boxes_sdf.
    manifest = json.loads(manifest_path.read_text())
    spawn_z = manifest["spawn_zone"]["elevation_m"] + 0.5

    record_video = LaunchConfiguration("record_video").perform(context)
    xacro_path = (
        FindPackageShare("regolith_rover_description").find("regolith_rover_description")
        + "/urdf/regolith_rover.urdf.xacro"
    )
    cine_camera = LaunchConfiguration("cine_camera").perform(context)
    urdf_xml = xacro.process_file(
        xacro_path, mappings={"record_video": record_video, "cine_camera": cine_camera}
    ).toxml()
    rover_urdf_path = output_dir / "rover.urdf"
    rover_urdf_path.write_text(urdf_xml)

    # Bake the rover into world.sdf at its spawn pose, alongside the rocks/terrain
    # already there - see _bake_rover_model_sdf's docstring for why. Must happen
    # before gz_sim (below) is actually started; IncludeLaunchDescription only holds
    # world_sdf_path as a string here, so rewriting the file now is safe - gz-sim
    # itself doesn't read it until the launch tree executes, after this function
    # returns.
    rover_model_sdf = _bake_rover_model_sdf(rover_urdf_path, spawn_z)
    world_sdf_path.write_text(
        world_sdf_path.read_text().replace("</world>", f"{rover_model_sdf}\n  </world>", 1)
    )

    # Cinematic camera rig - opt-in, off ("none") by default, so a normal run is
    # unaffected. See _cine_rig_sdf's docstring and camera_rig_node.py for the
    # mechanism. Spliced in the same way as the rover above, before gz_sim starts.
    cine_rig = LaunchConfiguration("cine_rig").perform(context)
    if cine_rig not in ("none", "", "false"):
        spawn_zone = manifest["spawn_zone"]
        # orbit needs a narrower hfov than chase/crane/track - see _cine_rig_sdf's
        # docstring: at the shared 1.3 rad the rover read as a ~3%-of-frame speck.
        rig_hfov = 0.7 if cine_rig == "orbit" else 1.3
        rig_sdf = _cine_rig_sdf(spawn_zone["x_m"], spawn_zone["y_m"], spawn_z, hfov=rig_hfov)
        world_sdf_path.write_text(
            world_sdf_path.read_text().replace("</world>", f"{rig_sdf}\n  </world>", 1)
        )

    headless = LaunchConfiguration("headless").perform(context)
    gz_flags = "-r -s" if headless.lower() == "true" else "-r"
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([FindPackageShare("ros_gz_sim"), "/launch/gz_sim.launch.py"]),
        launch_arguments={"gz_args": f"{gz_flags} {world_sdf_path}"}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": urdf_xml, "use_sim_time": True}],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist",
            "/odometry@nav_msgs/msg/Odometry@gz.msgs.Odometry",
            "/camera@sensor_msgs/msg/Image@gz.msgs.Image",
            "/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
            # Depth, for visual odometry. Shares camera_link and the RGB camera's
            # intrinsics, so /camera/camera_info describes both and pixel (u,v) is the
            # same ray in each - see regolith_rover.urdf.xacro.
            "/depth_camera@sensor_msgs/msg/Image@gz.msgs.Image",
            "/imu@sensor_msgs/msg/Imu@gz.msgs.IMU",
            f"/world/{WORLD_NAME}/model/{ROVER_NAME}/joint_state@sensor_msgs/msg/JointState@gz.msgs.Model",
            # Ground truth, comparison only - deliberately NOT fed into the EKF, and
            # NOT bridged to /tf (the EKF below is the sole publisher of odom -> base_link).
            f"/model/{ROVER_NAME}/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
        ],
        remappings=[
            ("/odometry", "/odom"),
            ("/camera", "/camera/image"),
            ("/camera_info", "/camera/camera_info"),
            ("/depth_camera", "/camera/depth"),
            (f"/world/{WORLD_NAME}/model/{ROVER_NAME}/joint_state", "/joint_states"),
            (f"/model/{ROVER_NAME}/pose", "/ground_truth/pose"),
        ],
    )

    # See PROGRESS.md M3 for why this relay is necessary (gz-sim publishes all-zero
    # sensor covariance, which robot_localization treats as "ignore this measurement";
    # the IMU's frame_id also needs fixing to match the URDF's TF tree).
    # Wheel-slip gate, upstream of the covariance relay: while the chassis is
    # pinned and the wheels spin, wheel odometry integrates distance that never
    # happened and the EKF - which has no absolute reference - keeps that error
    # forever (see wheel_slip_node.py and PROGRESS.md's res40 M4 failure). This
    # node republishes /odom as /odom/gated with a zero-velocity update
    # substituted while it detects slip, from onboard signals only.
    # A/B LEVER ONLY, default False (matches the shipped, fixed behaviour) - see
    # wheel_slip_node.py's "SIGNATURE 2, RETIRED" and PROGRESS.md. True restores the
    # pre-fix "rigid body" false-positive path for a same-build comparison campaign.
    legacy_rigid_body_signature = (
        LaunchConfiguration("legacy_rigid_body_signature").perform(context).lower() == "true"
    )
    wheel_slip_node = Node(
        package="regolith_bringup",
        executable="wheel_slip_node.py",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "legacy_rigid_body_signature": legacy_rigid_body_signature,
            }
        ],
    )

    sensor_covariance_relay = Node(
        package="regolith_bringup",
        executable="sensor_covariance_relay.py",
        output="screen",
        parameters=[{"use_sim_time": True, "odom_topic": "/odom/gated"}],
    )

    # Localization oracle, OFF by default and never part of a milestone result:
    # feeds the EKF a simulated absolute position reference (ground truth at
    # ~1 Hz, 0.5 m sigma) standing in for the visual odometry this PoC does not
    # have. It exists to test the M4 verification's falsifiable claim - that
    # localization is the only thing between this stack and the milestone. See
    # absolute_reference_relay.py and PROGRESS.md.
    # Visual odometry: the exteroceptive sensor M4's error budget identified as the
    # missing piece. It observes the lateral slide that a differential-drive model
    # cannot represent, and feeds the EKF vx/vy (see regolith_visual_odometry and
    # ekf.yaml's odom1). Cameras only, never ground truth - so unlike the oracle
    # below, runs with this on are legitimate milestone results.
    use_visual_odometry = LaunchConfiguration("visual_odometry").perform(context).lower() == "true"
    visual_odometry_node = Node(
        package="regolith_visual_odometry",
        executable="visual_odometry_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    if use_visual_odometry:
        print(
            "[hello_moon.launch] Visual odometry ENABLED - note this MEASURED WORSE than "
            "leaving it off, on all three M4 acceptance seeds (EKF divergence 0.4->1.4, "
            "0.7->24.7 and 6.1->15.2 m, same build). It is kept because the diagnosis is "
            "specific and fixable, not because it currently helps. See PROGRESS.md.",
            flush=True,
        )

    oracle = LaunchConfiguration("localization_oracle").perform(context).lower() == "true"
    terrain_relative = LaunchConfiguration("terrain_relative").perform(context).lower() == "true"
    if oracle and terrain_relative:
        raise RuntimeError(
            "localization_oracle and terrain_relative both publish /absolute_reference/pose "
            "and cannot run together. The oracle reads ground truth (experiments only); "
            "terrain_relative earns the same fix from the IMU and the a-priori DEM."
        )
    if oracle:
        ekf_config_name = "/config/ekf_oracle.yaml"
    elif terrain_relative:
        ekf_config_name = "/config/ekf_terrain_relative.yaml"
    else:
        ekf_config_name = "/config/ekf.yaml"
    ekf_config = FindPackageShare("regolith_bringup").find("regolith_bringup") + ekf_config_name
    if oracle:
        print(
            "[hello_moon.launch] LOCALIZATION ORACLE ENABLED - the EKF is being fed "
            "ground-truth position. This is an experiment; results are not milestone "
            "results. See PROGRESS.md.",
            flush=True,
        )
    if terrain_relative:
        print(
            "[hello_moon.launch] TERRAIN-RELATIVE NAVIGATION ENABLED - the EKF is fused with "
            "an absolute position fix matched from IMU attitude against the a-priori DEM. No "
            "ground truth is read, so runs with this on ARE milestone results. Replayed "
            "through 25 recorded runs it cuts final EKF error from 2.97 m to 0.71 m median, "
            "but a replay is not a run - see PROGRESS.md before reading anything into a "
            "single result.",
            flush=True,
        )
    ekf_node = Node(
        package="robot_localization",
        executable="ekf_node",
        name="ekf_filter_node",
        output="screen",
        parameters=[ekf_config],
    )

    absolute_reference_relay = Node(
        package="regolith_bringup",
        executable="absolute_reference_relay.py",
        output="screen",
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("localization_oracle")),
    )

    # Terrain-relative navigation: the earned version of the relay above. Reads
    # the same a-priori terrain manifest the costmap does - on a real mission,
    # an orbital DEM - and matches IMU attitude against it for an absolute fix.
    #
    # The dem_* arguments deliberately worsen that map. They default to zero (the
    # map as generated), and are the only way to run this stack on a map with a
    # mission's defects rather than on the generator's own heightmap read exactly.
    dem_post_m = float(LaunchConfiguration("dem_post_m").perform(context))
    dem_noise_m = float(LaunchConfiguration("dem_noise_m").perform(context))
    dem_shift_m = float(LaunchConfiguration("dem_shift_m").perform(context))
    dem_noise_seed = int(LaunchConfiguration("dem_noise_seed").perform(context))
    dem_prefilter_m = float(LaunchConfiguration("dem_prefilter_m").perform(context))
    if terrain_relative and (dem_post_m or dem_noise_m or dem_shift_m):
        print(
            f"[hello_moon.launch] A-PRIORI DEM DEGRADED: posts {dem_post_m} m, noise "
            f"{dem_noise_m} m, registration shift {dem_shift_m} m (seed {dem_noise_seed}). "
            "This run's localisation numbers are NOT perfect-map numbers.",
            flush=True,
        )
    terrain_relative_node = Node(
        package="regolith_bringup",
        executable="terrain_relative_node.py",
        output="screen",
        parameters=[{
            "manifest_path": str(manifest_path),
            "use_sim_time": True,
            "dem_post_m": dem_post_m,
            "dem_noise_m": dem_noise_m,
            "dem_shift_m": dem_shift_m,
            "dem_noise_seed": dem_noise_seed,
            "dem_prefilter_m": dem_prefilter_m,
        }],
        condition=IfCondition(LaunchConfiguration("terrain_relative")),
    )

    costmap_node = Node(
        package="regolith_costmap",
        executable="costmap_node",
        output="screen",
        parameters=[
            {
                "manifest_path": str(manifest_path),
                "resolution_m": 1.0,
                "rover_radius_m": 0.3,
                "slope_lethal_deg": 20.0,
                "use_sim_time": True,
            }
        ],
    )

    planner_node = Node(
        package="regolith_planner",
        executable="planner_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    # Exposed because it is the parameter M4's arrival error is most sensitive to:
    # the rover stops this far short of where it believes the goal is, straight off
    # a 1.5 m bar, before it has drifted at all. Overridable so the two settings can
    # be compared on the same build rather than argued about - see PROGRESS.md.
    goal_tolerance_m = float(LaunchConfiguration("goal_tolerance_m").perform(context))

    pure_pursuit_node = Node(
        package="regolith_vehicle_interface",
        executable="pure_pursuit_node",
        output="screen",
        parameters=[{"use_sim_time": True, "goal_tolerance_m": goal_tolerance_m}],
    )

    # Simulated flip recovery backstop: if the rover still ends up flipped (the
    # tilted-slab terrain collision makes this rare, not impossible), teleport it
    # upright to its last known-good pose via gz set_pose rather than leaving the
    # demo dead. Explicitly a simulated self-right - see flip_recovery_node.py.
    stuck_debug = LaunchConfiguration("stuck_debug").perform(context).lower() == "true"
    onboard_only_recovery = (
        LaunchConfiguration("onboard_only_recovery").perform(context).lower() == "true"
    )
    flip_recovery_node = Node(
        package="regolith_bringup",
        executable="flip_recovery_node.py",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "world_name": WORLD_NAME,
                "model_name": ROVER_NAME,
                "stuck_debug": stuck_debug,
                "onboard_only": onboard_only_recovery,
            }
        ],
    )

    tour_mission = Node(
        package="regolith_bringup",
        executable="tour_mission.py",
        output="screen",
        # The seed makes the route reproducible: the tour is drawn from the costmap, so
        # without it the same terrain would produce a different loop on every launch.
        parameters=[{"use_sim_time": True, "seed": seed}],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("mission"), "' == 'tour'"])
        ),
    )

    # Start/goal flags in Gazebo and markers in RViz. Purely a visualisation - it holds
    # no part of the control loop, and it is deliberately left OUT of the
    # shutdown-on-exit set below so a failed decoration cannot end a mission run.
    mission_markers = Node(
        package="regolith_bringup",
        executable="mission_markers_node.py",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "world_name": WORLD_NAME,
                "manifest_path": str(manifest_path),
            }
        ],
        condition=IfCondition(LaunchConfiguration("markers")),
    )

    # Cinematic camera rig driver - see _cine_rig_sdf above and camera_rig_node.py.
    # Opt-in (cine_rig defaults to "none"); excluded from shutdown_on_unexpected_exit
    # below for the same reason mission_markers is - a filming problem should never
    # end a mission run.
    camera_rig_node = Node(
        package="regolith_bringup",
        executable="camera_rig_node.py",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "world_name": WORLD_NAME,
                "rig_model_name": CINE_RIG_MODEL_NAME,
                "shot": cine_rig,
            }
        ],
        condition=IfCondition(
            PythonExpression(["'", LaunchConfiguration("cine_rig"), "' not in ('none', '', 'false')"])
        ),
    )

    rviz_config = (
        FindPackageShare("regolith_rover_description").find("regolith_rover_description")
        + "/rviz/rover.rviz"
    )
    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    # If any long-running node dies unexpectedly, shut the whole launch tree
    # down instead of leaving the rest running as orphans - a leftover node
    # set from a broken launch has no ROS_DOMAIN_ID/namespace isolation from a
    # later launch and silently fights it over shared topic names (this is
    # exactly what caused the overnight freeze investigated in PROGRESS.md:
    # `parameter_bridge` died, nothing tore the rest down, and a second launch
    # 8 minutes later collided with the survivors for the next 9 hours).
    shutdown_on_unexpected_exit = [
        RegisterEventHandler(
            OnProcessExit(
                target_action=node,
                on_exit=Shutdown(
                    reason=f"'{name}' exited unexpectedly - shutting down the rest of the demo"
                ),
            )
        )
        for name, node in [
            ("robot_state_publisher", robot_state_publisher),
            ("parameter_bridge", bridge),
            ("wheel_slip_node", wheel_slip_node),
            ("sensor_covariance_relay", sensor_covariance_relay),
            # If VO dies mid-run the rover silently reverts to the sensor suite that
            # scored 0/3, and the run would still report a number. Tear down instead.
            *([("visual_odometry_node", visual_odometry_node)] if use_visual_odometry else []),
            # Same reasoning as VO above: if the terrain matcher dies the run
            # silently continues on dead reckoning alone and still reports a
            # number, which would be a result attributed to the wrong stack.
            *([("terrain_relative_node", terrain_relative_node)] if terrain_relative else []),
            ("ekf_node", ekf_node),
            ("costmap_node", costmap_node),
            ("planner_node", planner_node),
            ("pure_pursuit_node", pure_pursuit_node),
            ("flip_recovery_node", flip_recovery_node),
            ("tour_mission", tour_mission),
            ("rviz2", rviz),
        ]
    ]

    return [
        gz_sim,
        robot_state_publisher,
        bridge,
        wheel_slip_node,
        absolute_reference_relay,
        terrain_relative_node,
        *([visual_odometry_node] if use_visual_odometry else []),
        sensor_covariance_relay,
        ekf_node,
        costmap_node,
        planner_node,
        pure_pursuit_node,
        flip_recovery_node,
        tour_mission,
        mission_markers,
        camera_rig_node,
        rviz,
        *shutdown_on_unexpected_exit,
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "seed", default_value="42", description="Terrain generation seed"
            ),
            DeclareLaunchArgument(
                "mission",
                default_value="none",
                description="'tour' runs a scripted 5-waypoint loop automatically; otherwise click goals in RViz",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="true",
                description="Launch RViz with the rover config (needed for clicking '2D Goal Pose' goals)",
            ),
            DeclareLaunchArgument(
                "markers",
                default_value="true",
                description="Flag the start point and the mission goals in Gazebo and RViz "
                "(mission_markers_node.py). Visual only - the flags have no "
                "collision and nothing in the control loop reads them",
            ),
            DeclareLaunchArgument(
                "localization_oracle",
                default_value="false",
                description="EXPERIMENT ONLY: feed the EKF a simulated absolute position "
                "reference from ground truth, standing in for visual odometry. "
                "Results obtained with this are not milestone results - see "
                "absolute_reference_relay.py",
            ),
            DeclareLaunchArgument(
                "terrain_relative",
                default_value="false",
                description="Terrain-relative navigation: an absolute x/y fix matched from IMU "
                "attitude against the a-priori terrain DEM (terrain_relative_node.py). "
                "Onboard sensors only, so unlike localization_oracle its results ARE "
                "milestone results. OFF by default pending live validation: replayed "
                "through 25 recorded runs it cuts final EKF error from 2.97 m to 0.71 m "
                "median, but a replay is not a run. See PROGRESS.md",
            ),
            DeclareLaunchArgument(
                "dem_post_m",
                default_value="0.0",
                description="Coarsen the a-priori DEM to this post spacing before matching "
                "(0 = as generated). Only affects terrain_relative. Replayed offline, "
                "detail loss degrades the fix GRACEFULLY - the matcher gets quieter "
                "rather than more wrong, because coarser terrain is more ambiguous and "
                "the margin gate rejects more windows. See PROGRESS.md",
            ),
            DeclareLaunchArgument(
                "dem_noise_m",
                default_value="0.0",
                description="Add spatially-correlated elevation error to the a-priori DEM, "
                "sigma in metres before smoothing (the node logs the realised rms). "
                "Only affects terrain_relative.",
            ),
            DeclareLaunchArgument(
                "dem_shift_m",
                default_value="0.0",
                description="Displace the a-priori DEM by this distance on both axes: "
                "registration error, the defect the offline replay says is FATAL. A 3 m "
                "offset made the fix worse than dead reckoning while publishing as many "
                "fixes at full confidence, because nothing about the cost surface is "
                "wrong - the map is simply not where it says it is. Only affects "
                "terrain_relative.",
            ),
            DeclareLaunchArgument(
                "dem_noise_seed",
                default_value="0",
                description="RNG seed for dem_noise_m, so a degraded map is reproducible.",
            ),
            DeclareLaunchArgument(
                "dem_prefilter_m",
                default_value="0.0",
                description="Smooth the a-priori DEM by this length BEFORE matching. Not a "
                "defect - the mitigation for one. The matcher reads slope, so height "
                "error of sigma correlated over L arrives as ~sigma/L of slope error, "
                "and lengthening L suppresses it at the same rate it costs signal. In "
                "replay this recovers 0.1 m of DEM vertical error from 6.16 m to 1.73 m "
                "median, but costs 0.71 -> 1.66 m on a map that was already good. OFF by "
                "default for that reason: it pays only when the DEM's vertical accuracy "
                "is known to be poor.",
            ),
            DeclareLaunchArgument(
                "visual_odometry",
                default_value="false",
                description="RGB-D visual odometry feeding body-frame vy to the EKF. OFF by "
                "default because a controlled comparison measured it making "
                "localization WORSE on all three acceptance seeds (EKF divergence "
                "0.4->1.4, 0.7->24.7 and 6.1->15.2 m, same build, oracle off). It is "
                "a real onboard sensor, not an oracle, so its results would be "
                "milestone results - they are just bad ones. See PROGRESS.md",
            ),
            DeclareLaunchArgument(
                "goal_tolerance_m",
                default_value="0.35",
                description="How close pure pursuit drives to the commanded goal before "
                "stopping. Counts against M4's 1.5 m arrival bar directly, so "
                "it is a measured setting, not a taste one. Tightened from the "
                "original 1.0 m after the seed-7 replicate campaign showed a "
                "clean 0/3 vs 3/3 separation with no orbiting fallback firing - "
                "see PROGRESS.md, 'The stopping tolerance, measured'",
            ),
            DeclareLaunchArgument(
                "legacy_rigid_body_signature",
                default_value="false",
                description="A/B LEVER ONLY. Re-enables wheel_slip_node's retired 'signature "
                "2' (attitude-span + gyro-RMS rigid-body check), which false-positives on "
                "ordinary straight-line driving over smooth ground - see PROGRESS.md, "
                "'Root-caused: the benign-ground traction stall was never a stall'. Default "
                "false matches the shipped, fixed behaviour; true is for a same-build "
                "before/after comparison campaign only, not a setting to ship on",
            ),
            DeclareLaunchArgument(
                "stuck_debug",
                default_value="false",
                description="DIAGNOSTIC ONLY. flip_recovery_node logs every _stuck_since "
                "streak start/reset/fire at its own 5 Hz tick, with the instantaneous "
                "gt_speed/commanded_speed that caused it. For investigating the fixed-arm "
                "chokepoint split (PROGRESS.md) - not a setting to run campaigns with by "
                "default, it is verbose",
            ),
            DeclareLaunchArgument(
                "headless",
                default_value="false",
                description="Run Gazebo server-only (-s), no GUI window - for unattended/automated runs",
            ),
            DeclareLaunchArgument(
                "onboard_only_recovery",
                default_value="false",
                description="SIM-TO-REAL. Makes flip/stuck recovery read only what a real "
                "rover has - IMU attitude and the EKF estimate - instead of "
                "/ground_truth/pose, and stops it teleporting the rover upright on a flip. "
                "Ground truth is still logged beside every verdict for scoring. Default "
                "false: the shipped numbers were all measured with the oracle",
            ),
            DeclareLaunchArgument(
                "cine_camera",
                default_value="none",
                description="MEDIA CAPTURE ONLY. Adds a second, recording-only camera to the "
                "rover - 'onboard' (mast over the deck, chassis and front wheels in frame) or "
                "'chase' (boom behind and above, whole rover in frame). Leaves the onboard "
                "perception cameras untouched. Record it via the /rover/cine/record_video "
                "gz service - see regolith_bringup's README",
            ),
            DeclareLaunchArgument(
                "cine_rig",
                default_value="none",
                description="MEDIA CAPTURE ONLY. Adds a free-flying camera rig, decoupled from "
                "the rover (no joint, no shared link, does not touch its mass/inertia/"
                "collisions/sensors) and driven each tick via gz-sim's set_pose service by "
                "camera_rig_node.py - 'chase' (smoothed follow, behind and above), 'orbit' "
                "(slow circle around the rover), 'crane' (low/close reveal that rises and "
                "pulls back), or 'track' (fixed lockoff, camera pans to follow the rover past "
                "it). Default 'none': off, no extra camera, no extra node. Record it via the "
                "/rover/rig/record_video gz service, same pattern as cine_camera - see "
                "regolith_bringup's README",
            ),
            DeclareLaunchArgument(
                "cine_light",
                default_value="false",
                description="MEDIA CAPTURE ONLY. Raises the sun to 32 degrees and lifts the "
                "scene ambient so surface texture is legible on video. Render-only - terrain, "
                "rocks and costmap are unchanged",
            ),
            DeclareLaunchArgument(
                "record_video",
                default_value="false",
                description="If true, adds gz-sim's CameraVideoRecorder plugin to the onboard "
                "camera - see regolith_rover.urdf.xacro for how to start/stop it",
            ),
            OpaqueFunction(function=_generate_and_launch),
        ]
    )
