"""
pixhawk_tcp_3dstream.py
Streams Pixhawk 2.4.8 position (x,y,z) and attitude (roll,pitch,yaw)
to a TCP client (Processing 3D visualizer).
"""

import time
import socket
from pymavlink import mavutil

# --- CONFIG ---
PIXHAWK_PORT = "/dev/ttyACM0"  # Change as needed (e.g. COM3 on Windows)
BAUD = 57600
TCP_IP = "127.0.0.1"
TCP_PORT = 5760
SEND_HZ = 30
# ---------------


def connect_pixhawk(port, baud):
    print(f"[INFO] Connecting to Pixhawk on {port} @ {baud} baud...")
    mav = mavutil.mavlink_connection(port, baud=baud)
    mav.wait_heartbeat()
    print("[OK] Heartbeat received.")
    return mav


def main():
    mav = connect_pixhawk(PIXHAWK_PORT, BAUD)

    # TCP Server
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_IP, TCP_PORT))
    server.listen(1)
    print(f"[INFO] Waiting for Processing to connect on {TCP_IP}:{TCP_PORT}...")
    conn, addr = server.accept()
    print(f"[OK] Connected by {addr}")

    last_x = last_y = last_z = 0
    last_roll = last_pitch = last_yaw = 0
    period = 1.0 / SEND_HZ

    try:
        while True:
            # Fetch latest messages (non-blocking)
            msg = mav.recv_match(
                type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False
            )
            if msg:
                if msg.get_type() == "LOCAL_POSITION_NED":
                    last_x, last_y, last_z = msg.x, msg.y, msg.z
                elif msg.get_type() == "ATTITUDE":
                    last_roll, last_pitch, last_yaw = msg.roll, msg.pitch, msg.yaw

            # Send both position + attitude
            line = f"{last_x:.3f},{last_y:.3f},{last_z:.3f},{last_roll:.3f},{last_pitch:.3f},{last_yaw:.3f}\n"
            conn.sendall(line.encode("utf-8"))

            time.sleep(period)

    except (BrokenPipeError, ConnectionResetError):
        print("[WARN] Connection lost.")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        conn.close()
        server.close()
        mav.close()


if __name__ == "__main__":
    main()
