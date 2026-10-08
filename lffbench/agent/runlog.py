"""Run directory: config.json, events.jsonl (one JSON object per line), images/, summary.json."""
import json
import os
import time

import numpy as np

from .images import encode_jpeg, resize


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "__dataclass_fields__"):
        return {k: getattr(o, k) for k in o.__dataclass_fields__}
    return str(o)


def dumps(obj):
    return json.dumps(obj, default=_default)


class RunLogger:
    def __init__(self, run_dir, save_images=True, jpeg_quality=90):
        self.run_dir = run_dir
        self.save_images = save_images
        self.jpeg_quality = jpeg_quality
        os.makedirs(os.path.join(run_dir, "images"), exist_ok=True)
        self.f = open(os.path.join(run_dir, "events.jsonl"), "a")
        self.t0 = time.time()

    def event(self, type, **fields):
        rec = dict(type=type, t=round(time.time() - self.t0, 3), **fields)
        self.f.write(dumps(rec) + "\n")
        self.f.flush()
        return rec

    def write_json(self, name, obj):
        with open(os.path.join(self.run_dir, name), "w") as f:
            json.dump(obj, f, indent=1, default=_default)

    def save_image(self, ref, rel, res=None):
        """Encode once (this exact JPEG is what the model receives) and save it under images/."""
        if res is not None:
            ref.array = resize(ref.array, res)
        ref.jpeg = encode_jpeg(ref.array, self.jpeg_quality)
        if self.save_images:
            path = os.path.join(self.run_dir, "images", rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(ref.jpeg)
            ref.path = os.path.relpath(path, self.run_dir)
        return ref

    def close(self):
        self.f.close()
