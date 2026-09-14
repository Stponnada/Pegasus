#!/usr/bin/env python3
"""Bridge local TCP ports through SSH and a Slurm job using standard I/O."""

from __future__ import annotations

import argparse
import os
import selectors
import shlex
import socket
import socketserver
import subprocess
import sys
import threading

READY_MARKER = b"__ONTOMEM_BRIDGE_READY__\n"


def connect(host: str, port: int) -> None:
    with socket.create_connection((host, port)) as peer:
        selector = selectors.DefaultSelector()
        selector.register(sys.stdin.fileno(), selectors.EVENT_READ, "stdin")
        selector.register(peer, selectors.EVENT_READ, "socket")
        stdin_open = True

        while True:
            for key, _ in selector.select():
                if key.data == "stdin":
                    data = os.read(sys.stdin.fileno(), 65536)
                    if data:
                        peer.sendall(data)
                    elif stdin_open:
                        stdin_open = False
                        selector.unregister(sys.stdin.fileno())
                        peer.shutdown(socket.SHUT_WR)
                else:
                    data = peer.recv(65536)
                    if not data:
                        return
                    view = memoryview(data)
                    while view:
                        view = view[os.write(sys.stdout.fileno(), view) :]


class TunnelHandler(socketserver.BaseRequestHandler):
    remote = ""
    job_id = ""
    node = ""
    destination_port = 0
    remote_python = ""
    remote_bridge = ""

    def handle(self) -> None:
        slurm_command = [
            "srun",
            f"--jobid={self.job_id}",
            "--overlap",
            "--nodes=1",
            "--ntasks=1",
            f"--nodelist={self.node}",
            "--cpus-per-task=1",
            "--mem=64M",
            self.remote_python,
            "-u",
            self.remote_bridge,
            "connect",
            "--host=127.0.0.1",
            f"--port={self.destination_port}",
        ]
        remote_command = (
            f"printf {shlex.quote(READY_MARKER.decode())}; "
            f"exec {shlex.join(slurm_command)}"
        )
        process = subprocess.Popen(
            # ServerAliveInterval/CountMax: a non-streaming completion request
            # can hold this connection completely silent (zero bytes either
            # direction) for several minutes while the model generates. With
            # no keepalive, an idle-connection timeout somewhere on the
            # university network (a firewall/NAT between the Mac and the
            # cluster, not anything in this script) silently drops the TCP
            # connection with no error surfaced here -- confirmed via vLLM's
            # own logs showing the request completing normally server-side
            # while the client saw "Remote end closed connection without
            # response". Sending an SSH-level keepalive every 30s (up to 10
            # missed replies, ~5 min grace) keeps the connection looking
            # active to any such intermediate device during long, silent
            # generations.
            ["ssh", "-T", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10", self.remote, remote_command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert process.stdin is not None
        assert process.stdout is not None

        try:
            preamble = bytearray()
            while READY_MARKER not in preamble:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    raise ConnectionError("SSH bridge closed before becoming ready")
                preamble.extend(chunk)
                if len(preamble) > 131072:
                    raise ConnectionError("SSH bridge emitted an oversized preamble")

            remainder = bytes(preamble).split(READY_MARKER, 1)[1]
            if remainder:
                self.request.sendall(remainder)

            upload = threading.Thread(
                target=self._copy_client_to_process,
                args=(process,),
                daemon=True,
            )
            upload.start()
            while data := process.stdout.read1(65536):
                self.request.sendall(data)
        except (BrokenPipeError, ConnectionError, OSError) as exc:
            print(f"bridge connection closed: {exc}", file=sys.stderr)
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _copy_client_to_process(self, process: subprocess.Popen[bytes]) -> None:
        assert process.stdin is not None
        try:
            while data := self.request.recv(65536):
                process.stdin.write(data)
                process.stdin.flush()
            process.stdin.close()
        except (BrokenPipeError, ConnectionError, OSError):
            pass


class ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


def handler_for(
    remote: str,
    job_id: str,
    node: str,
    destination_port: int,
    remote_python: str,
    remote_bridge: str,
) -> type[TunnelHandler]:
    class ConfiguredTunnelHandler(TunnelHandler):
        pass

    ConfiguredTunnelHandler.remote = remote
    ConfiguredTunnelHandler.job_id = job_id
    ConfiguredTunnelHandler.node = node
    ConfiguredTunnelHandler.destination_port = destination_port
    ConfiguredTunnelHandler.remote_python = remote_python
    ConfiguredTunnelHandler.remote_bridge = remote_bridge
    return ConfiguredTunnelHandler


def listen(args: argparse.Namespace) -> None:
    specifications = (
        (args.local_llm_port, args.llm_node, args.llm_port),
        (args.local_embed_port, args.embed_node, args.embed_port),
    )
    servers = [
        ThreadingTCPServer(
            ("127.0.0.1", local_port),
            handler_for(
                args.remote,
                args.job_id,
                node,
                destination_port,
                args.remote_python,
                args.remote_bridge,
            ),
        )
        for local_port, node, destination_port in specifications
    ]
    threads = [
        threading.Thread(target=server.serve_forever, daemon=True)
        for server in servers
    ]
    for thread in threads:
        thread.start()

    print(
        f"Forwarding job {args.job_id}: "
        f"localhost:{args.local_llm_port} -> {args.llm_node}:{args.llm_port}, "
        f"localhost:{args.local_embed_port} -> "
        f"{args.embed_node}:{args.embed_port}"
    )
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for server in servers:
            server.shutdown()
            server.server_close()


def listen_one(args: argparse.Namespace) -> None:
    server = ThreadingTCPServer(
        ("127.0.0.1", args.local_port),
        handler_for(
            args.remote,
            args.job_id,
            args.node,
            args.destination_port,
            args.remote_python,
            args.remote_bridge,
        ),
    )
    print(
        f"Forwarding job {args.job_id}: localhost:{args.local_port} -> "
        f"{args.node}:{args.destination_port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    connect_parser = subparsers.add_parser("connect")
    connect_parser.add_argument("--host", required=True)
    connect_parser.add_argument("--port", required=True, type=int)

    listen_parser = subparsers.add_parser("listen")
    listen_parser.add_argument("--remote", required=True)
    listen_parser.add_argument("--job-id", required=True)
    listen_parser.add_argument("--llm-node", required=True)
    listen_parser.add_argument("--embed-node", required=True)
    listen_parser.add_argument("--llm-port", required=True, type=int)
    listen_parser.add_argument("--embed-port", required=True, type=int)
    listen_parser.add_argument("--local-llm-port", required=True, type=int)
    listen_parser.add_argument("--local-embed-port", required=True, type=int)
    listen_parser.add_argument("--remote-python", required=True)
    listen_parser.add_argument("--remote-bridge", required=True)

    listen_one_parser = subparsers.add_parser("listen-one")
    listen_one_parser.add_argument("--remote", required=True)
    listen_one_parser.add_argument("--job-id", required=True)
    listen_one_parser.add_argument("--node", required=True)
    listen_one_parser.add_argument("--destination-port", required=True, type=int)
    listen_one_parser.add_argument("--local-port", required=True, type=int)
    listen_one_parser.add_argument("--remote-python", required=True)
    listen_one_parser.add_argument("--remote-bridge", required=True)

    args = parser.parse_args()
    if args.command == "connect":
        connect(args.host, args.port)
    elif args.command == "listen-one":
        listen_one(args)
    else:
        listen(args)


if __name__ == "__main__":
    main()
