# UAS Autonomous Position Stabilisation Stack

Cascade PID flight controller for 3D position hold and yaw control of a DJI Tello quadcopter, 
validated in PyBullet simulation with mean positional error < 0.01 m.

## Architecture
- **Outer loop** — position → desired velocity
- **Inner loop** — velocity → RC command
- **Body-frame transform** — 2D rotation matrix on GPS error
- **Yaw priority gate** — freezes translation if heading error > 0.25 rad
- **Anti-windup integral clamping** — rejects constant wind disturbances
- **CSV logger** — records 3D error, RPY attitude, and target deviation

## Stack
Python 3 · NumPy · PyBullet · DJI Tello SDK

## Performance
| Metric | Target |
|---|---|
| Mean position error | < 0.01 m |
| Mean yaw error | < 0.01 rad |

## Module
AERO60492 — Autonomous Mobile Robots, MSc Robotics, University of Manchester
