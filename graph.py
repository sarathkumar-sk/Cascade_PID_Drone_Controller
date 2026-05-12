import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the flight log
df = pd.read_csv('sim_flight_log_2.csv')
df['time_s'] = df['dt_s'].cumsum()

# Apply a modern dark theme
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Flight Simulation Analysis Dashboard', fontsize=20, color='#00e6e6', fontweight='bold')

# 1. 3D Trajectory Visualization with Highlighted Target Points
ax1 = fig.add_subplot(2, 2, 1, projection='3d')

# Plot Actual Path (Cyan line)
ax1.plot(df['pos_x'], df['pos_y'], df['pos_z'], label='Actual Path', color='#00ffcc', linewidth=2, zorder=5)

# Plot Target Path (Dashed pink line)
ax1.plot(df['tgt_x'], df['tgt_y'], df['tgt_z'], color='#ff007f', linestyle='--', alpha=0.3)

# HIGHLIGHT: Scatter plot for Target Points (Pink dots with white edge)
ax1.scatter(df['tgt_x'], df['tgt_y'], df['tgt_z'], 
            color='#ff007f', s=25, alpha=0.9, label='Target Points', 
            marker='o', edgecolors='white', linewidths=0.5)

ax1.set_title('3D Flight Trajectory', fontsize=14, pad=20)

# MODERNIZATION: Add padding to axis labels to avoid overlap with scale numbers
ax1.set_xlabel('X (m)', labelpad=12)
ax1.set_ylabel('Y (m)', labelpad=12)
ax1.set_zlabel('Z (m)', labelpad=12)

# MODERNIZATION: Move legend outside the plotting area
ax1.legend(loc='upper left', bbox_to_anchor=(1.1, 0.9), frameon=False)

ax1.grid(color='gray', linestyle=':', alpha=0.2)
ax1.view_init(elev=25, azim=-45) # Set a clean perspective angle

# 2. 3D Position Error Over Time
ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(df['time_s'], df['pos_error_3d'], color='#ffff00', linewidth=2)
ax2.fill_between(df['time_s'], df['pos_error_3d'], color='#ffff00', alpha=0.15)
ax2.set_title('Total 3D Error Magnitude', fontsize=14)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Error (m)')
ax2.grid(color='gray', linestyle='--', alpha=0.2)

# 3. Attitude (Roll, Pitch, Yaw)
ax3 = fig.add_subplot(2, 2, 3)
colors = ['#00ff00', '#ff8c00', '#1e90ff']
for i, col in enumerate(['roll', 'pitch', 'yaw']):
    ax3.plot(df['time_s'], df[col], label=col.capitalize(), color=colors[i], alpha=0.9)
ax3.set_title('Attitude Dynamics', fontsize=14)
ax3.set_xlabel('Time (seconds)')
ax3.set_ylabel('Degrees')
ax3.legend(loc='upper right', frameon=False)
ax3.grid(color='gray', linestyle='--', alpha=0.2)

# 4. Coordinate-wise Errors
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(df['time_s'], df['err_x_world'], label='X Error', color='#ff4d4d')
ax4.plot(df['time_s'], df['err_y_world'], label='Y Error', color='#4dff4d')
ax4.plot(df['time_s'], df['err_z'], label='Z Error', color='#4d4dff')
ax4.set_title('Individual Axis Errors', fontsize=14)
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Error (m)')
ax4.legend(ncol=3, frameon=False)
ax4.grid(color='gray', linestyle='--', alpha=0.2)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()