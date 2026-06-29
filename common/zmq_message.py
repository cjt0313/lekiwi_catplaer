"""Standard ZMQ message format with JSON serialization."""

import json
import time
from dataclasses import asdict, dataclass

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



def make_publisher(address: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.bind(address)
    return sock


def publish(sock: zmq.Socket, msg: ZmqMessage):
    topic = msg.header.msg_type.encode("utf-8")
    sock.send_multipart([topic, msg.to_bytes()])


