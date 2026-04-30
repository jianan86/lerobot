import os
import pickle  # nosec
import struct
import sys
import time


def _read_exact(num_bytes: int) -> bytes:
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sys.stdin.buffer.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _configure_qt_fontdir() -> None:
    if os.environ.get("QT_QPA_FONTDIR"):
        return

    candidates = (
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts",
    )
    for path in candidates:
        if os.path.isdir(path):
            os.environ["QT_QPA_FONTDIR"] = path
            return


def main() -> int:
    _configure_qt_fontdir()
    import cv2
    _configure_qt_fontdir()

    window_name = sys.argv[1] if len(sys.argv) > 1 else "pose_act_observation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        try:
            header = _read_exact(4)
        except EOFError:
            break
        payload_size = struct.unpack("!I", header)[0]
        payload = _read_exact(payload_size)
        frame = pickle.loads(payload)  # nosec
        cv2.imshow(window_name, frame)
        cv2.pollKey()
        time.sleep(0.01)

    try:
        cv2.destroyWindow(window_name)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
