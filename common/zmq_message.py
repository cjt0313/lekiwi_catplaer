"""Standard ZMQ message format with JSON serialization."""

import json
import time
from dataclasses import dataclass, asdict

import zmq

from common.types import MsgType


@dataclass
class ZmqHeader:
    msg_type: str
    stamp: float
    frame_id: str
    seq: int
    source: str


@dataclass
class ZmqMessage:
    header: ZmqHeader
    payload: dict

    @classmethod
    def create(cls, msg_type: MsgType, payload: dict, source: str,
               seq: int = 0, frame_id: str = "map") -> "ZmqMessage":
        header = ZmqHeader(
            msg_type=msg_type.value,
            stamp=time.time(),
            frame_id=frame_id,
            seq=seq,
            source=source,
        )
        return cls(header=header, payload=payload)

    def to_bytes(self) -> bytes:
        data = {
            "header": asdict(self.header),
            "payload": self.payload,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "ZmqMessage":
        data = json.loads(raw.decode("utf-8"))
        header = ZmqHeader(**data["header"])
        return cls(header=header, payload=data["payload"])


def make_publisher(address: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(address)
    return sock


def make_subscriber(address: str, topics: list[str] | None = None) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(address)
    if topics:
        for t in topics:
            sock.setsockopt_string(zmq.SUBSCRIBE, t)
    else:
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
    return sock


def publish(sock: zmq.Socket, msg: ZmqMessage):
    topic = msg.header.msg_type.encode("utf-8")
    sock.send_multipart([topic, msg.to_bytes()])


def receive(sock: zmq.Socket, timeout_ms: int = 100) -> ZmqMessage | None:
    if sock.poll(timeout_ms):
        parts = sock.recv_multipart()
        if len(parts) == 2:
            return ZmqMessage.from_bytes(parts[1])
    return None
