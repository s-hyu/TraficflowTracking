import numpy as np
import cv2
import torch
import torch.nn.functional as F

try:
    from fastreid.config import get_cfg
    from fastreid.modeling import build_model
    from fastreid.utils.checkpoint import Checkpointer
    from fastreid.data.transforms import build_transforms
    _FASTREID_AVAILABLE = True
except Exception:
    _FASTREID_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


class FastReIDExtractor(object):
    def __init__(self, cfg_file, weights, device="cuda", batch_size=32):
        if not _FASTREID_AVAILABLE:
            raise ImportError("fastreid is not available. Please install FastReID and its dependencies.")
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow is required for FastReID preprocessing.")
        if not cfg_file or not weights:
            raise ValueError("FastReID requires cfg_file and weights.")
        cfg = get_cfg()
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.WEIGHTS = weights
        cfg.MODEL.DEVICE = device
        cfg.freeze()

        self.model = build_model(cfg)
        self.model.eval()
        self.model.to(device)
        Checkpointer(self.model).load(weights)

        self.transform = build_transforms(cfg, is_train=False)
        self.device = device
        self.batch_size = batch_size

    @staticmethod
    def _clip_boxes(boxes, w, h):
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)
        return boxes

    def extract(self, image_bgr, tlbrs):
        if tlbrs is None or len(tlbrs) == 0:
            return np.zeros((0, 0), dtype=np.float32)

        h, w = image_bgr.shape[:2]
        boxes = np.asarray(tlbrs, dtype=np.float32).copy()
        boxes = self._clip_boxes(boxes, w, h)
        patches = []
        for x1, y1, x2, y2 in boxes:
            if x2 <= x1 or y2 <= y1:
                patches.append(None)
                continue
            patch = image_bgr[int(y1):int(y2), int(x1):int(x2)]
            patches.append(patch)

        feats = []
        with torch.no_grad():
            for i in range(0, len(patches), self.batch_size):
                batch_patches = []
                valid_idx = []
                for j, patch in enumerate(patches[i:i + self.batch_size]):
                    if patch is None or patch.size == 0:
                        batch_patches.append(None)
                        continue
                    img = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(img)
                    tensor = self.transform(pil)
                    batch_patches.append(tensor)
                    valid_idx.append(j)

                if not valid_idx:
                    feats.extend([np.zeros((0,), dtype=np.float32) for _ in batch_patches])
                    continue

                batch = torch.stack([t for t in batch_patches if t is not None], dim=0).to(self.device)
                outputs = self.model(batch)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
                outputs = F.normalize(outputs, dim=1)
                outputs = outputs.detach().cpu().numpy()

                out_iter = iter(outputs)
                for t in batch_patches:
                    if t is None:
                        feats.append(np.zeros((outputs.shape[1],), dtype=np.float32))
                    else:
                        feats.append(next(out_iter))

        return np.asarray(feats, dtype=np.float32)
