# -*- coding: utf-8 -*-
"""WiLoR 多候选输出：每个手框都按“左手”“右手”两种假设各跑一次，给时序跟踪 / 双目关联用。

``WiLoRBackend.predict`` 每帧只留一只左手一只右手（handedness 由 YOLO 决定）。手部跟踪和
双目互相纠正需要看到全部候选：

* 原图 YOLO 框 + 水平翻转图上的 YOLO 框（翻回原图坐标，左右标签互换）合并去重，每个框记两次检测的
  左右投票和分数（翻转测试时增强，TTA）。
* 每个框用 right=1 和 right=0 各跑一次 WiLoR。WiLoR 处理左手的方式就是把 crop 水平翻转后当右手跑，
  所以 “right=0 跑原图框” 与 “把整张图翻转后按右手跑这个框再翻回” 是同一个计算——这就是模型侧的
  翻转 TTA：两种假设的结果都留着，由后面的跟踪 / 双目几何来选。
* ``predict_boxes`` 允许外部给框（双目互相纠正：把另一目的 3D 估计投过来再裁一次）。
"""
import numpy as np

from headcam.hand_pose import JOINTS, WiLoRBackend, _full_camera_from_batch, recursive_to


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_boxes(orig, flipped, iou_thr=0.5):
    """orig / flipped：[(box, is_right, score)]（flipped 已翻回原图坐标并互换左右）。
    返回 [{"box", "votes": {"orig": (is_right, score) or None, "flip": ...}}]。"""
    out = []
    for box, r, s in orig:
        out.append({"box": list(map(float, box)), "orig": [float(r), float(s)], "flip": None})
    for box, r, s in flipped:
        best, bi = 0.0, -1
        for i, c in enumerate(out):
            v = _iou(c["box"], box)
            if v > best:
                best, bi = v, i
        if best >= iou_thr and out[bi]["flip"] is None:
            out[bi]["flip"] = [float(r), float(s)]
        else:
            out.append({"box": list(map(float, box)), "orig": None, "flip": [float(r), float(s)]})
    return out


class WiLoRCandidates(object):
    def __init__(self, backend=None, det_conf=0.2):
        self.backend = backend or WiLoRBackend()
        self.det_conf = float(det_conf)

    def _yolo(self, bgr):
        res = self.backend._detector(bgr, conf=self.det_conf, verbose=False)[0]
        out = []
        for d in res:
            box = d.boxes.data.cpu().detach().squeeze().numpy()
            out.append((box[:4].tolist(), float(d.boxes.cls.cpu().item()), float(d.boxes.conf.cpu().item())))
        return out

    def detect(self, image_rgb, flip_tta=True):
        import cv2
        b = self.backend
        b._ensure()
        bgr = cv2.cvtColor(np.ascontiguousarray(image_rgb), cv2.COLOR_RGB2BGR)
        orig = self._yolo(bgr)
        flipped = []
        if flip_tta:
            w = bgr.shape[1]
            for box, r, s in self._yolo(np.ascontiguousarray(bgr[:, ::-1])):
                flipped.append(([w - box[2], box[1], w - box[0], box[3]], 1.0 - r, s))
        return merge_boxes(orig, flipped), bgr

    def predict_boxes(self, bgr, boxes, rights, calib=None):
        """每个 (box, right) 跑一次 WiLoR。返回 [{"keypoints_2d", "joints_cam"}]。"""
        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        b = self.backend
        b._ensure()
        if not boxes:
            return []
        ds = ViTDetDataset(b._cfg, bgr, np.asarray(boxes, dtype=float), np.asarray(rights, dtype=float),
                           rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        res = []
        for batch in loader:
            batch = recursive_to(batch, b._device)
            with torch.no_grad():
                out = b._model(batch)
            mult = (2 * batch["right"] - 1)
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = mult * pred_cam[:, 1]
            cam_metric, cam_pixels, focal = _full_camera_from_batch(pred_cam, batch, b._cfg, calib)
            j3 = out["pred_keypoints_3d"].detach().cpu().numpy()
            rr = batch["right"].detach().cpu().numpy()
            wh = batch["img_size"].detach().cpu().numpy()
            for i in range(j3.shape[0]):
                shape = (int(wh[i][1]), int(wh[i][0]))
                jc, kp = b._camera_joints(j3[i], rr[i], cam_metric[i], shape,
                                          project_translation=cam_pixels[i], project_focal=focal)
                res.append({"keypoints_2d": np.asarray(kp, float)[:JOINTS, :2].tolist(),
                            "joints_cam": np.asarray(jc, float)[:JOINTS].tolist()})
        return res

    def frame_candidates(self, image_rgb, calib=None, flip_tta=True, both_hypotheses=True):
        cands, bgr = self.detect(image_rgb, flip_tta=flip_tta)
        boxes, rights, owner = [], [], []
        for i, c in enumerate(cands):
            vote = c["orig"] or c["flip"]
            hyps = (1.0, 0.0) if both_hypotheses else (vote[0],)
            for r in hyps:
                boxes.append(c["box"]); rights.append(r); owner.append((i, r))
        preds = self.predict_boxes(bgr, boxes, rights, calib)
        for (i, r), p in zip(owner, preds):
            cands[i]["right" if r >= 0.5 else "left"] = p
        return cands
