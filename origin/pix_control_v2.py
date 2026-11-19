import time
import socket
from pymavlink import mavutil

PIXHAWK_PORT = "/dev/ttyACM0"
BAUD = 57600
TCP_IP = "127.0.0.1"
TCP_PORT = 5760
SEND_HZ = 30


def connect_pixhawk(port, baud):
    print(f"[INFO] Connecting to Pixhawk on {port} @ {baud} baud...")
    mav = mavutil.mavlink_connection(port, baud=baud)
    mav.wait_heartbeat()
    print("[OK] Heartbeat received.")
    return mav


def reset_hitl(mav):
    """Send a reset to re-zero HITL origin."""
    print("[INFO] Resetting HITL origin...")
    try:
        # Option 1: reset attitude and position via command
        mav.mav.command_long_send(
            mav.target_system,
            mav.target_component,
            mavutil.mavlink.MAV_CMD_PREFLIGHT_REBOOT_SHUTDOWN,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        print("[OK] Sent reboot command to Pixhawk.")
    except Exception as e:
        print(f"[WARN] Reset failed: {e}")


def main():
    mav = connect_pixhawk(PIXHAWK_PORT, BAUD)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((TCP_IP, TCP_PORT))
    server.listen(1)
    print(f"[INFO] Waiting for Processing to connect on {TCP_IP}:{TCP_PORT}...")
    conn, addr = server.accept()
    print(f"[OK] Connected by {addr}")
    conn.setblocking(False)

    last_x = last_y = last_z = 0
    last_roll = last_pitch = last_yaw = 0
    period = 1.0 / SEND_HZ
    buffer = b""

    while True:
        # 🔹 read possible RESET command
        try:
            data = conn.recv(1024)
            if data:
                buffer += data
                if b"\n" in buffer:
                    lines = buffer.split(b"\n")
                    for line in lines[:-1]:
                        cmd = line.decode().strip()
                        if cmd == "RESET":
                            print("[INFO] Reset command received from Processing.")
                            reset_hitl(mav)
                            conn.sendall(b"ACK\n")
                    buffer = lines[-1]
        except BlockingIOError:
            pass

        msg = mav.recv_match(type=["LOCAL_POSITION_NED", "ATTITUDE"], blocking=False)
        if msg:
            if msg.get_type() == "LOCAL_POSITION_NED":
                last_x, last_y, last_z = msg.x, msg.y, msg.z
            elif msg.get_type() == "ATTITUDE":
                last_roll, last_pitch, last_yaw = msg.roll, msg.pitch, msg.yaw

        line = f"{last_x:.3f},{last_y:.3f},{last_z:.3f},{last_roll:.3f},{last_pitch:.3f},{last_yaw:.3f}\n"
        try:
            conn.sendall(line.encode("utf-8"))
        except (BrokenPipeError, ConnectionResetError):
            print("[WARN] Connection lost. Waiting for reconnect...")
            conn, addr = server.accept()
            conn.setblocking(False)
            print(f"[OK] Reconnected by {addr}")

        time.sleep(period)


if __name__ == "__main__":
    main()
