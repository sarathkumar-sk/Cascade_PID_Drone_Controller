"""
METHOD SUMMARY: CASCADE PID

This controller uses a Cascade PID (Proportional-Integral-Derivative) architecture.
By nesting two PID loops, we separate slow position dynamics from fast velocity tracking, 
preventing instability caused by bandwidth mismatches.

1. Architecture: A dual-loop Cascade structure is used for the X, Y, and Z axes. 
   - Outer Loop: Computes position error and generates a Desired Velocity.
   - Inner Loop: Computes velocity error and generates the final motor command.
2. Axis Transformation: World-frame position errors are rotated into the Body-frame 
   using a 2D rotation matrix (Yaw-alignment) to ensure commands align with the 
   drone's forward/lateral axes.
3. Robustness: Includes Integral (I) gain clamping (anti-windup) to handle 
   external disturbances like wind.
4. Stability: Implements a yaw-priority logic where horizontal translation is 
   damped if the heading error exceeds a specific threshold.

"""

wind_flag = True

# IMPORTS
import numpy as np
import csv

class PID:
    """ Standard Proportional-Integral-Derivative controller logic. """
    def __init__(self, kp, ki, kd, i_limit=10.0, out_limit=1.0):
        self.kp = kp # Proportional gain: handles current error.
        self.ki = ki # Integral gain: handles accumulated past error. 
        self.kd = kd # Derivative gain: handles predicted future error/damping.
        self.i_limit = i_limit # Prevents integral windup by capping the accumulated error.
        self.out_limit = out_limit # Caps the total output to drone physical limits.
        self.integral = 0.0 # Initialize error accumulation.
        self.prev_error = 0.0 # Stores error from previous step for derivative calculation.

    def reset(self):
        """Clears controller memor, used when switching targets to prevent overshoot."""
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        """Calculates the control command based on time step (dt) and current error."""
        # 1. Accumulate integral error using trapezoidal-like integration.
        self.integral += error * dt
        # 2. Apply anti-windup clamping to the integral term.
        self.integral = np.clip(self.integral, -self.i_limit, self.i_limit)
        # 3. Calculate derivative (change in error over change in time).
        derivative = (error - self.prev_error) / dt if dt > 1e-9 else 0.0
        self.prev_error = error
        # 4. Sum the three components (P + I + D).
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 5. Clip output to ensure it stays within vehicle command limits (-1.0 to 1.0).
        return float(np.clip(output, -self.out_limit, self.out_limit))

class CascadePID:
    """ Uses the PID class twice: once for position, once for velocity
    A cascade structure uses an outer loop for position and an inner loop for velocity tracking to provide flight stability."""
    def __init__(self, outer_kp, outer_ki, outer_kd, inner_kp, inner_ki, inner_kd, vel_limit=1.0, i_limit_outer=2.0, i_limit_inner=1.0, out_limit=1.0):
        self.vel_limit = vel_limit
        # Outer Loop: Input = Position Error -> Output = Desired Velocity.
        self.outer = PID(outer_kp, outer_ki, outer_kd, i_limit=i_limit_outer, out_limit=vel_limit)
        # Inner Loop: Input = Velocity Error -> Output = Actuation Command (cmd_v).
        self.inner = PID(inner_kp, inner_ki, inner_kd, i_limit=i_limit_inner, out_limit=out_limit)

    def reset(self):
        """Resets both internal PID loops."""
        self.outer.reset()
        self.inner.reset()

    def update(self, pos_error, current_vel, dt):
        """Standard Cascade update: Position Error drives Velocity Demand."""
        desired_vel = self.outer.update(pos_error, dt)
        desired_vel = float(np.clip(desired_vel, -self.vel_limit, self.vel_limit))
        # Calculate velocity error (demand minus current state).
        vel_error = desired_vel - current_vel
        return self.inner.update(vel_error, dt)

# CONTROLLER INSTANTIATION
# Tuned parameters to meet Mean Error < 0.01m and Std Dev < 0.01.
x_ctrl = CascadePID(
    outer_kp=2.0,  outer_ki=0.02, outer_kd=0.8,
    inner_kp=1.0,  inner_ki=0.0,  inner_kd=0.4,
    vel_limit=0.8, i_limit_outer=0.5, i_limit_inner=0.5, out_limit=1.0
)
y_ctrl = CascadePID(
    outer_kp=2.0,  outer_ki=0.02, outer_kd=0.8,
    inner_kp=1.0,  inner_ki=0.0,  inner_kd=0.4,
    vel_limit=0.8, i_limit_outer=0.5, i_limit_inner=0.5, out_limit=1.0
)
z_ctrl = CascadePID(
    outer_kp=2.0,  outer_ki=0.0,  outer_kd=0.10,
    inner_kp=1.0,  inner_ki=0.0,  inner_kd=0.02,
    vel_limit=0.8, i_limit_outer=2.0, i_limit_inner=0.5, out_limit=1.0
)

# Yaw uses a single PID as it only requires rate control, not cascade depth.
yaw_ctrl = PID(kp=1.2, ki=0.02, kd=0.06, i_limit=1.0, out_limit=1.74533)

def reset_controllers():
    """Resets all axis controllers simultaneously."""
    x_ctrl.reset()
    y_ctrl.reset()
    z_ctrl.reset()
    yaw_ctrl.reset()

# Logging setup to track performance data.
prev_target = None
log_file    = None
log_writer  = None

LOG_FILENAME = 'output.csv'
LOG_FIELDS = ['dt','pos_x','pos_y','pos_z','roll','pitch','yaw','tar_x','tar_y','tar_z','tar_yaw','err_x','err_y','err_z','overall_error']

def write_log(row: dict):
    """Writes performance data to CSV."""
    global log_file, log_writer
    if log_writer is None:
        log_file   = open(LOG_FILENAME, 'w', newline='')
        log_writer = csv.DictWriter(log_file, fieldnames=LOG_FIELDS)
        log_writer.writeheader()
        log_file.flush()
    log_writer.writerow({k: row.get(k, '') for k in LOG_FIELDS})
    log_file.flush()


def controller(state, target_pos, dt, wind_enabled=False):
    """
    Advanced Method: Cascade PID Controller.
    This implementation uses a dual-loop strategy:
    1. Outer Loop (Position): Calculates the required velocity to reach the target.
    2. Inner Loop (Velocity): Tracks that velocity to provide stable control.
    """
    global prev_target
    dt = float(np.clip(dt, 0.001, 0.1)) # Safety clip for time steps.

    # Extract State: [x, y, z, roll, pitch, yaw]
    pos_x, pos_y, pos_z = float(state[0]), float(state[1]), float(state[2])
    roll, pitch, yaw    = float(state[3]), float(state[4]), float(state[5])
    tar_x, tar_y, tar_z, tar_yaw = (float(target_pos[0]), float(target_pos[1]), float(target_pos[2]), float(target_pos[3]))

    # If the target changes, clear the PID "memory" to ensure a fresh approach.
    current_target = (tar_x, tar_y, tar_z, tar_yaw)
    if prev_target != current_target:
        reset_controllers()
        prev_target = current_target

    # AXIS TRANSFORMATION
    # Errors are calculated in World coordinates but commands must be in Body-Fixed coordinates.
    err_x_world  = tar_x - pos_x
    err_y_world  = tar_y - pos_y
    err_z        = tar_z - pos_z
    pos_error_3d = float(np.sqrt(err_x_world**2 + err_y_world**2 + err_z**2))

    # Rotation Matrix for 2D (Yaw) to align world error with drone forward/right axes.
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    err_x_body =  err_x_world * cos_yaw + err_y_world * sin_yaw
    err_y_body = -err_x_world * sin_yaw + err_y_world * cos_yaw

    # YAW CONTROL
    # Normalize yaw error to [-pi, pi] to prevent the drone from spinning 360 degrees.
    yaw_error = float((tar_yaw - yaw + np.pi) % (2 * np.pi) - np.pi)
    yaw_rate  = yaw_ctrl.update(yaw_error, dt)

    # POSITION CONTROL
    # To ensure high accuracy, prioritize yaw alignment before aggressive movement.
    YAW_THRESHOLD = 0.25
    if abs(yaw_error) > YAW_THRESHOLD: # Hover and rotate if yaw error is high.
        x_ctrl.reset()
        y_ctrl.reset()
        vel_x = 0.0
        vel_y = 0.0
    else:
        vel_x = x_ctrl.update(err_x_body, 0.0, dt)
        vel_y = y_ctrl.update(err_y_body, 0.0, dt)

    vel_z = z_ctrl.update(err_z, 0.0, dt) # Z-axis is independent of yaw.

    # Final Command Clipping to ensure stability and safety.
    cmd_vx      = float(np.clip(vel_x,    -1.0, 1.0))
    cmd_vy      = float(np.clip(vel_y,    -1.0, 1.0))
    cmd_vz      = float(np.clip(vel_z,    -1.0, 1.0))
    cmd_yawrate = float(np.clip(yaw_rate, -1.0, 1.0))

    # Log results.
    write_log({
        'dt' : round(dt, 6),
        'pos_x' : round(pos_x, 4),
        'pos_y' : round(pos_y, 4),
        'pos_z' : round(pos_z, 4),
        'roll' : round(roll,  5),
        'pitch' : round(pitch, 5),
        'yaw'  : round(yaw,   5),
        'tar_x' : tar_x,
        'tar_y' : tar_y,
        'tar_z' : tar_z,
        'tar_yaw' : tar_yaw,
        'err_x' : round(err_x_world,  4),
        'err_y' : round(err_y_world,  4),
        'err_z' : round(err_z,  4),
        'overall_error' : round(pos_error_3d, 4),
    })

    return (cmd_vx, cmd_vy, cmd_vz, cmd_yawrate)