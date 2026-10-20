"""stdio <-> Unix socket relay: the MCP "server" a jailed harness starts (docs/15 §3.4).

The real MCP server (scripts/mcp_robot_server.py) runs outside the jail, attached to the other end of the socket by
lffbench/harnesses/jail.py; this relay only copies bytes, so the jail needs no repo code and no venv. Standard library
only (runs on the system python inside the jail).
  python3 mcp_fwd.py /tmp/lffsock/<hex>/mcp.sock
"""
import os
import socket
import sys
import threading


def main():
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(sys.argv[1])

    def upstream():
        try:
            while True:
                b = os.read(0, 65536)
                if not b:
                    break
                s.sendall(b)
        finally:
            try:
                s.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    threading.Thread(target=upstream, daemon=True).start()
    while True:
        b = s.recv(65536)
        if not b:
            break
        mv = memoryview(b)
        while mv:
            n = os.write(1, mv)
            mv = mv[n:]


if __name__ == "__main__":
    main()
