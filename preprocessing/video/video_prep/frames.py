import json
import subprocess
from pathlib import Path
import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_type,codec_name,width,height,r_frame_rate,sample_aspect_ratio,duration"
                          ":format=duration", "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    v = next(s for s in d["streams"] if s["codec_type"] == "video")
    num, den = v["r_frame_rate"].split("/")
    sar = v.get("sample_aspect_ratio", "1:1")
    duration = float(v.get("duration") or d["format"]["duration"])
    return {"width": v["width"], "height": v["height"], "codec": v["codec_name"],
            "fps": round(int(num) / int(den), 3), "duration": round(duration, 3),
            "sar": "1:1" if sar in ("N/A", "0:1") else sar,
            "has_audio": any(s["codec_type"] == "audio" for s in d["streams"])}


def iter_frames(path, size, fps, gray=False, crop=None, scale=1.0, start=0.0, end=None):

    w, h = size
    vf = [f"fps={fps}:round=up"]
    if crop:
        x0, y0, x1, y1 = crop
        vf.append(f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0}")
        w, h = x1 - x0, y1 - y0
    if scale != 1.0:
        w, h = round(w * scale), round(h * scale)
        vf.append(f"scale={w}:{h}:flags=area")
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start > 0:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(path)]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vf", ",".join(vf), "-f", "rawvideo", "-pix_fmt", "gray" if gray else "rgb24", "pipe:1"]
    ch = 1 if gray else 3
    n = w * h * ch
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    try:
        k = 0
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            arr = np.frombuffer(buf, np.uint8).reshape((h, w) if gray else (h, w, 3))
            yield round(start + k / fps, 3), arr
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def grab_frame(path, t, fps, out_png):
    out_png = Path(out_png)
    out_png.unlink(missing_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(path),
                    "-ss", f"{max(0.0, t - 1 / fps + 0.001):.3f}", "-frames:v", "1", str(out_png)], check=True)
    if not out_png.exists():
        raise RuntimeError(f"{t}s 프레임을 뽑지 못함: {path}")
