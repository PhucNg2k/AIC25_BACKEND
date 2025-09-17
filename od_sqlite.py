from pathlib import Path
import sqlite3
from typing import Dict, List, Tuple, Optional, Union, Any
import math

ImageResult = Dict[str, object]  # loose alias


def parse_count_condition(expr: Optional[str]) -> Tuple[str, int]:
    if not expr:
        return ('>', 0)
    expr = expr.strip().replace(' ', '')
    for op in ('>=', '<=', '==', '>', '<'):
        if expr.startswith(op):
            try:
                return (op, int(expr[len(op):])) # return tuple of operation and integer (">=", 3)
            except Exception:
                return ('>', 0)
    try:
        return ('>=', int(expr))
    except Exception:
        return ('>', 0)


def eval_count(op: str, lhs: int, rhs: int) -> bool:
    if op == '>=': return lhs >= rhs
    if op == '>': return lhs > rhs
    if op == '==': return lhs == rhs
    if op == '<=': return lhs <= rhs
    if op == '<': return lhs < rhs
    return False


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a['x'], a['y'], a['w'], a['h']
    bx1, by1, bw, bh = b['x'], b['y'], b['w'], b['h']
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0: return 0.0
    aarea = max(0, aw) * max(0, ah)
    barea = max(0, bw) * max(0, bh)
    denom = aarea + barea - inter
    return inter / denom if denom > 0 else 0.0


class ODSearcher:
    """
    Walks *.sqlite shards and queries frames/detections/grid/counts
    according to your schema.
    """
    def __init__(self, root_dir: str, grid_rows: int = 8, grid_cols: int = 8):
        self.root_dir = Path(root_dir)
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.db_paths = sorted(self.root_dir.rglob('*.sqlite'))

    # --- helpers
    def _cells_from_bbox(self, box: Dict[str, int]) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        W, H = 1920, 1080
        cx1, cy1, w, h = box['x'], box['y'], box['w'], box['h']
        cx2, cy2 = cx1 + w, cy1 + h
        ty1 = H - (cy1 + h)
        ty2 = H - cy1
        x1, x2 = max(0, min(cx1, W)), max(0, min(cx2, W))
        y1, y2 = max(0, min(ty1, H)), max(0, min(ty2, H))
        cell_w, cell_h = W / self.grid_cols, H / self.grid_rows
        c0, c1 = int(x1 // cell_w), int((x2 - 1e-6) // cell_w)
        r0, r1 = int(y1 // cell_h), int((y2 - 1e-6) // cell_h)
        c0, c1 = max(0, min(self.grid_cols-1, c0)), max(0, min(self.grid_cols-1, c1))
        r0, r1 = max(0, min(self.grid_rows-1, r0)), max(0, min(self.grid_rows-1, r1))
        return (r0, r1), (c0, c1)

    # --- queries
    def _frames_meeting_count(self, con, label: str, cond: str) -> List[int]:
        op, rhs = parse_count_condition(cond)
        cur = con.cursor()
        cur.execute("SELECT frame_id, count FROM object_counts WHERE label_unified = ?", (label,))
        return [fid for fid, cnt in cur.fetchall() if eval_count(op, cnt, rhs)]

    def _frames_meeting_position(self, con, label: str, cell_rows, cell_cols) -> List[int]:
        (r0, r1), (c0, c1) = cell_rows, cell_cols
        cur = con.cursor()
        cur.execute("""
            SELECT DISTINCT d.frame_id
            FROM detection_grid g
            JOIN detections d ON d.id = g.detection_id
            WHERE d.label_unified = ?
              AND g.grid_rows = ? AND g.grid_cols = ?
              AND g.cell_row BETWEEN ? AND ?
              AND g.cell_col BETWEEN ? AND ?
        """, (label, self.grid_rows, self.grid_cols, r0, r1, c0, c1))
        return [row[0] for row in cur.fetchall()]

    def _detections_in_bbox(self, con, label: str, frame_id: int, bbox):
        cur = con.cursor()
        cur.execute("SELECT id, x1, y1, w, h, score FROM detections WHERE frame_id = ? AND label_unified = ?",
                    (frame_id, label))
        matches = []
        for det_id, x1, y1, w, h, score in cur.fetchall():
            iou = iou_xywh({'x': x1, 'y': y1, 'w': w, 'h': h}, bbox)
            if iou > 0.05:
                matches.append({'det_id': det_id, 'score': float(score), 'iou': float(iou)})
        return matches

    def _frame_meta(self, con, frame_id: int):
        cur = con.cursor()
        cur.execute("SELECT video_id, frame_idx, COALESCE(time_sec, -1.0), path FROM frames WHERE id = ?", (frame_id,))
        row = cur.fetchone()
        return (row[0], row[1], float(row[2]), row[3]) if row else None

    # --- main
    def search(self, obj_mask: Dict[str, Union[dict, object]]) -> List[ImageResult]:
        """
        Accepts obj_mask where each value can be:
          - a dict: {"count_condition": str, "bbox": [ {x,y,w,h}, ... ]}
          - a Pydantic ClassMask with bbox as list of BBox models
        We normalize everything to plain dicts for the SQL logic below.
        """

        def _bbox_to_dict(b: Any) -> Dict[str, int]:
            # Handles BBox model, dict, or any object with x/y/w/h attributes
            if isinstance(b, dict):
                return {"x": int(b["x"]), "y": int(b["y"]), "w": int(b["w"]), "h": int(b["h"])}
            if hasattr(b, "model_dump"):   # Pydantic v2
                d = b.model_dump()
                return {"x": int(d["x"]), "y": int(d["y"]), "w": int(d["w"]), "h": int(d["h"])}
            if hasattr(b, "dict"):         # Pydantic v1 fallback
                d = b.dict()
                return {"x": int(d["x"]), "y": int(d["y"]), "w": int(d["w"]), "h": int(d["h"])}
            # Generic object with attributes
            return {"x": int(getattr(b, "x")), "y": int(getattr(b, "y")),
                    "w": int(getattr(b, "w")), "h": int(getattr(b, "h"))}

        def _payload_to_dict(p: Any) -> Dict[str, Any]:
            # dict payload
            if isinstance(p, dict):
                bbs = [ _bbox_to_dict(bb) for bb in (p.get("bbox") or []) ]
                return {"count_condition": p.get("count_condition", ">0"), "bbox": bbs}
            # Pydantic ClassMask (or lookalike)
            if hasattr(p, "count_condition") and hasattr(p, "bbox"):
                cond = getattr(p, "count_condition", ">0") or ">0"
                bbs_raw = getattr(p, "bbox", []) or []
                bbs = [ _bbox_to_dict(bb) for bb in bbs_raw ]
                return {"count_condition": cond, "bbox": bbs}
            # Unknown type → default
            return {"count_condition": ">0", "bbox": []}

        results: List[ImageResult] = []

        for db_path in self.db_paths:
            print(db_path)
            con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=3.0)
            try:
                candidate_frames = None

                # -------- first pass: build candidate frame set
                for cls, raw_payload in obj_mask.items():
                    payload = _payload_to_dict(raw_payload)
                    # print(f"cls = {cls}, payload = {payload}")
                    cond = payload["count_condition"]
                    bboxes = payload["bbox"]
                    # print(f"COND = {cond}")
                    # print(f"bboxes = {bboxes}")
                    frames_by_count = set(self._frames_meeting_count(con, cls, cond))
                    print(f"frames_by_count = {frames_by_count}")
                    
                    if bboxes:
                        pos_union = set()
                        for nb in bboxes:
                            rows, cols = self._cells_from_bbox(nb)
                            pos_union |= set(self._frames_meeting_position(con, cls, rows, cols))
                        frames_ok = frames_by_count & pos_union if pos_union else frames_by_count
                    else:
                        frames_ok = frames_by_count

                    candidate_frames = frames_ok if candidate_frames is None else (candidate_frames & frames_ok)

                if not candidate_frames:
                    continue
                # print(candidate_frames)
                # -------- second pass: score frames
                for frame_id in candidate_frames:
                    meta = self._frame_meta(con, frame_id)
                    if not meta:
                        continue
                    video_id, frame_idx, pts_time, path = meta
                    frame_score = 0.0

                    for cls, raw_payload in obj_mask.items():
                        payload = _payload_to_dict(raw_payload)
                        bboxes = payload["bbox"]

                        if bboxes:
                            best = 0.0
                            for nb in bboxes:
                                matches = self._detections_in_bbox(con, cls, frame_id, nb)
                                if matches:
                                    best = max(best, max(m["score"] * (0.5 + 0.5 * m["iou"]) for m in matches))
                            frame_score += best
                        else:
                            cur = con.cursor()
                            cur.execute(
                                "SELECT MAX(score) FROM detections WHERE frame_id = ? AND label_unified = ?",
                                (frame_id, cls),
                            )
                            row = cur.fetchone()
                            frame_score += float(row[0] or 0.0)

                    results.append({
                        "video_name": video_id,
                        "frame_idx": int(frame_idx),
                        "pts_time": float(pts_time),
                        "image_path": path,
                        "score": float(frame_score),
                    })
            finally:
                con.close()

        # Deduplicate per (video_name, frame_idx)
        best_by_key: Dict[tuple, ImageResult] = {}
        for r in results:
            key = (r["video_name"], r["frame_idx"])
            if key not in best_by_key or r["score"] > best_by_key[key]["score"]:
                best_by_key[key] = r
        return list(best_by_key.values())