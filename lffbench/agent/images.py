"""Image references and encoding (JPEG bytes are what the model sees and what is saved to disk)."""
import base64
import io
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class ImageRef:
    camera: str  # agent-facing camera name ("agentview" / "wrist")
    array: Any = None  # uint8 HxWx3
    label: str = ""
    path: Optional[str] = None  # where the encoded image was saved
    jpeg: Optional[bytes] = None  # encoded bytes actually sent to the model

    def encoded(self, quality=90):
        if self.jpeg is None:
            if self.array is None and self.path:
                with open(self.path, "rb") as f:
                    self.jpeg = f.read()
            else:
                self.jpeg = encode_jpeg(self.array, quality)
        return self.jpeg

    def data_url(self, quality=90):
        return "data:image/jpeg;base64," + base64.b64encode(self.encoded(quality)).decode("ascii")

    @property
    def size(self):
        return None if self.array is None else tuple(np.asarray(self.array).shape[:2])


def encode_jpeg(arr, quality=90):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


def resize(arr, res):
    from PIL import Image
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape[0] == res and arr.shape[1] == res:
        return arr
    return np.asarray(Image.fromarray(arr).resize((res, res), Image.BILINEAR))
