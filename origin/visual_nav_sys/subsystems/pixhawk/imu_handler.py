"""
Pixhawk IMU Handler
Thread-safe Pixhawk IMU data reader using pymavlink
"""

import time
import numpy as np
from pymavlink import mavutil
from threading import Thread, Lock


class PixhawkIMU:
    """Thread-safe Pixhawk IMU data reader using pymavlink"""
    
    def __init__(self, connection_string='udp:127.0.0.1:14550'):
        """
        Initialize connection to Pixhawk
        connection_string options:
        - Serial: '/dev/ttyACM0' or 'COM3' (adjust based on your system)
        - UDP: 'udp:127.0.0.1:14550' (for SITL or MAVProxy forwarding)
        - TCP: 'tcp:127.0.0.1:5760'
        """
        self.connection_string = connection_string
        self.master = None
        self.lock = Lock()
        
        # IMU data (raw)
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.rollspeed = 0.0
        self.pitchspeed = 0.0
        self.yawspeed = 0.0
        
        # Acceleration (m/s^2) - raw
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        
        # EKF2 data (more stable, filtered estimates)
        self.ekf_pos_x = 0.0
        self.ekf_pos_y = 0.0
        self.ekf_pos_z = 0.0
        self.ekf_vel_x = 0.0
        self.ekf_vel_y = 0.0
        self.ekf_vel_z = 0.0
        
        # EKF status
        self.ekf_flags = 0
        self.ekf_healthy = False
        
        # Linear velocity estimation (integrated from accel) - for comparison
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        
        # Status
        self.connected = False
        self.running = False
        self.last_update = time.time()
        self.last_ekf_update = time.time()
        
    def connect(self):
        """Connect to Pixhawk with high-frequency data streams"""
        try:
            print(f"Connecting to Pixhawk at {self.connection_string}...")
            self.master = mavutil.mavlink_connection(self.connection_string, baud=115200)
            
            # Wait for heartbeat
            print("Waiting for heartbeat...")
            self.master.wait_heartbeat(timeout=10)
            print(f"✓ Heartbeat from system {self.master.target_system}")
            
            # Request HIGH FREQUENCY data streams (50Hz)
            # Request ALL streams at high rate
            for stream_id in [
                mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
                mavutil.mavlink.MAV_DATA_STREAM_EXTENDED_STATUS,
                mavutil.mavlink.MAV_DATA_STREAM_RC_CHANNELS,
                mavutil.mavlink.MAV_DATA_STREAM_POSITION,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1,  # ATTITUDE
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA2,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA3,
            ]:
                self.master.mav.request_data_stream_send(
                    self.master.target_system,
                    self.master.target_component,
                    stream_id,
                    50,  # 50 Hz - FAST!
                    1    # Start
                )
            
            # Also use MAVLink2 SET_MESSAGE_INTERVAL for specific high-rate messages
            # ATTITUDE (30) at 100Hz
            self._request_message_interval(30, 10000)  # 10000us = 100Hz
            # RAW_IMU (27) at 100Hz  
            self._request_message_interval(27, 10000)
            # LOCAL_POSITION_NED (32) at 50Hz
            self._request_message_interval(32, 20000)
            
            self.connected = True
            print("✓ High-frequency data streams requested (50-100Hz)")
            return True
            
        except Exception as e:
            print(f"✗ Failed to connect to Pixhawk: {e}")
            print("  Make sure Pixhawk is connected and MAVLink is available")
            print("  For USB: Try /dev/ttyACM0 or COM port")
            print("  For simulation: Use udp:127.0.0.1:14550")
            return False
    
    def _request_message_interval(self, message_id, interval_us):
        """Request specific message at given interval (MAVLink2)"""
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,  # confirmation
                message_id,  # Message ID
                interval_us,  # Interval in microseconds
                0, 0, 0, 0, 0  # Unused params
            )
        except:
            pass  # Older firmware may not support this
    
    def start(self):
        """Start IMU data reading thread"""
        if not self.connected:
            print("Cannot start - not connected to Pixhawk")
            return False
        
        self.running = True
        self.thread = Thread(target=self._update_loop, daemon=True)
        self.thread.start()
        print("✓ IMU reading thread started")
        return True
    
    def _update_loop(self):
        """Ultra-fast continuous update loop for IMU data"""
        last_time = time.perf_counter()
        msg_count = 0
        
        while self.running:
            try:
                # Batch read - get multiple messages per iteration
                for _ in range(10):  # Read up to 10 messages per loop
                    msg = self.master.recv_match(blocking=False)
                    if msg is None:
                        break
                    
                    msg_type = msg.get_type()
                    current_time = time.perf_counter()
                    dt = current_time - last_time
                    
                    # Direct attribute access (no lock for read-heavy operations)
                    if msg_type == 'ATTITUDE':
                        self.roll = msg.roll
                        self.pitch = msg.pitch
                        self.yaw = msg.yaw
                        self.rollspeed = msg.rollspeed
                        self.pitchspeed = msg.pitchspeed
                        self.yawspeed = msg.yawspeed
                        self.last_update = time.time()
                        self.connected = True
                        msg_count += 1
                    
                    elif msg_type == 'RAW_IMU':
                        # Convert from raw units to m/s^2
                        self.accel_x = msg.xacc / 1000.0 * 9.81
                        self.accel_y = msg.yacc / 1000.0 * 9.81
                        self.accel_z = msg.zacc / 1000.0 * 9.81
                        
                        # Fast velocity integration
                        if dt < 0.1:
                            self.vel_x += self.accel_x * dt
                            self.vel_y += self.accel_y * dt
                            self.vel_z += self.accel_z * dt
                            # Decay to prevent runaway
                            self.vel_x *= 0.99
                            self.vel_y *= 0.99
                            self.vel_z *= 0.99
                    
                    elif msg_type == 'LOCAL_POSITION_NED':
                        self.ekf_pos_x = msg.x
                        self.ekf_pos_y = msg.y
                        self.ekf_pos_z = msg.z
                        self.ekf_vel_x = msg.vx
                        self.ekf_vel_y = msg.vy
                        self.ekf_vel_z = msg.vz
                        self.last_ekf_update = time.time()
                        self.ekf_healthy = True
                    
                    elif msg_type == 'EKF_STATUS_REPORT':
                        self.ekf_flags = msg.flags
                        self.ekf_healthy = (msg.flags & 0x001) != 0
                    
                    elif msg_type == 'HEARTBEAT':
                        self.connected = True
                    
                    last_time = current_time
                
                # Minimal sleep - just yield to other threads
                time.sleep(0.0001)  # 0.1ms
                
            except Exception as e:
                if self.running:
                    print(f"IMU read error: {e}")
                time.sleep(0.01)
    
    def get_attitude(self):
        """Get current attitude (thread-safe)"""
        with self.lock:
            return {
                'roll': self.roll,
                'pitch': self.pitch,
                'yaw': self.yaw,
                'rollspeed': self.rollspeed,
                'pitchspeed': self.pitchspeed,
                'yawspeed': self.yawspeed,
                'connected': self.connected,
                'age': time.time() - self.last_update
            }
    
    def get_acceleration(self):
        """Get current acceleration (thread-safe)"""
        with self.lock:
            return {
                'x': self.accel_x,
                'y': self.accel_y,
                'z': self.accel_z,
                'vel_x': self.vel_x,
                'vel_y': self.vel_y,
                'vel_z': self.vel_z
            }
    
    def get_ekf_data(self):
        """Get EKF filtered data (thread-safe)"""
        with self.lock:
            return {
                'pos_x': self.ekf_pos_x,
                'pos_y': self.ekf_pos_y,
                'pos_z': self.ekf_pos_z,
                'vel_x': self.ekf_vel_x,
                'vel_y': self.ekf_vel_y,
                'vel_z': self.ekf_vel_z,
                'healthy': self.ekf_healthy,
                'flags': self.ekf_flags,
                'age': time.time() - self.last_ekf_update
            }
    
    def reset_velocity(self):
        """Reset integrated velocity (useful when AUV stops)"""
        with self.lock:
            self.vel_x = 0.0
            self.vel_y = 0.0
            self.vel_z = 0.0
    
    def stop(self):
        """Stop IMU reading"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if self.master:
            self.master.close()
        print("IMU reader stopped")
