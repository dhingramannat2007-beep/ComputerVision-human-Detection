"""
test_tracking.py

Large-scale test script for human detection + tracking.
Run on any video file to verify:
  - Tracking IDs stay consistent (same person = same ID)
  - Unique human count is accurate
  - No ghost detections or ID switches

USAGE:
    python test_tracking.py --video path/to/video.mp4
    python test_tracking.py --video 0           # webcam
    python test_tracking.py --video path/to/video.mp4 --save output.mp4
"""

import os
import time
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------- CONFIG ----------
CONF_THRESH         = 0.40
TRACKER             = "bytetrack.yaml"
RECOG_EVERY_N_FRAMES = 5
DET_SIZE            = (640, 640)
NEW_HUMAN_THRESHOLD  = 0.75
KNOWN_MATCH_THRESHOLD = 0.65
KNOWN_DIR           = "known_faces"

# ---------- HELPERS (same as main code) ----------

def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

def cosine_distance(a, b):
    return float(1.0 - np.dot(a, b))

def face_center_in_box(face_bbox, person_bbox):
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)

def load_known_faces(face_app, known_dir):
    if not os.path.isdir(known_dir):
        return {}
    known = {}
    person_dirs = [d for d in os.listdir(known_dir)
                   if os.path.isdir(os.path.join(known_dir, d))]
    for person in sorted(person_dirs):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            import glob
            imgs.extend(glob.glob(os.path.join(known_dir, person, ext)))
        embs = []
        for p in sorted(imgs):
            img = cv2.imread(p)
            if img is None:
                continue
            faces = face_app.get(img)
            if not faces:
                continue
            best = max(faces, key=lambda f: f.det_score)
            embs.append(l2_normalize(best.embedding.astype(np.float32)))
        if embs:
            avg = l2_normalize(np.mean(np.stack(embs), axis=0))
            known[person] = avg
            print(f"[OK] Known identity loaded: {person} ({len(embs)} images)")
    return known

def match_known_name(face_emb_norm, known_db):
    if not known_db:
        return None, 999.0
    best_name, best_dist = None, 999.0
    for name, kemb in known_db.items():
        d = cosine_distance(face_emb_norm, kemb)
        if d < best_dist:
            best_dist = d
            best_name = name
    if best_dist <= KNOWN_MATCH_THRESHOLD:
        return best_name, best_dist
    return None, best_dist


# ---------- TEST STATS TRACKER ----------

class TrackingStats:
    def __init__(self):
        self.frame_counts       = []   # how many people in each frame
        self.id_switch_events   = []   # (frame, tid, old_hid, new_hid)
        self.unique_humans_over_time = []  # unique count at each frame
        self.max_in_frame       = 0
        self.total_frames       = 0
        self.prev_tid_to_human  = {}

    def update(self, frame_idx, current_tids, tracker_to_human, humans):
        count = len(current_tids)
        self.frame_counts.append(count)
        self.unique_humans_over_time.append(len(humans))
        self.max_in_frame = max(self.max_in_frame, count)
        self.total_frames += 1

        # Detect ID switches: same tracker ID mapped to a different human ID
        for tid in current_tids:
            hid = tracker_to_human.get(tid)
            if hid is None:
                continue
            prev_hid = self.prev_tid_to_human.get(tid)
            if prev_hid is not None and prev_hid != hid:
                self.id_switch_events.append((frame_idx, tid, prev_hid, hid))
            self.prev_tid_to_human[tid] = hid

    def print_summary(self, elapsed_total):
        print("\n" + "="*55)
        print("           TRACKING TEST RESULTS")
        print("="*55)
        print(f"  Total frames processed   : {self.total_frames}")
        print(f"  Total time               : {elapsed_total:.1f}s")
        avg_fps = self.total_frames / elapsed_total if elapsed_total > 0 else 0
        print(f"  Average FPS              : {avg_fps:.1f}")
        print()
        print(f"  Unique humans detected   : {self.unique_humans_over_time[-1] if self.unique_humans_over_time else 0}")
        print(f"  Max people in one frame  : {self.max_in_frame}")
        avg_count = np.mean(self.frame_counts) if self.frame_counts else 0
        print(f"  Avg people per frame     : {avg_count:.2f}")
        print()
        print(f"  ID switches detected     : {len(self.id_switch_events)}")
        if self.id_switch_events:
            print("  (An ID switch = tracker gave same person a new human ID)")
            print("  ID switch log (first 10):")
            for frame_idx, tid, old_hid, new_hid in self.id_switch_events[:10]:
                print(f"    Frame {frame_idx}: Tracker {tid} switched Human {old_hid} -> Human {new_hid}")
        print()

        # Verdict
        switches = len(self.id_switch_events)
        if switches == 0:
            print("  ✅ TRACKING: Perfect — no ID switches!")
        elif switches <= 3:
            print(f"  ⚠️  TRACKING: Good — only {switches} ID switch(es), mostly stable")
        else:
            print(f"  ❌ TRACKING: {switches} ID switches — consider tuning thresholds")

        print("="*55)


# ---------- MAIN TEST ----------

def main():
    parser = argparse.ArgumentParser(description="Test human tracking at scale")
    parser.add_argument("--video", default="0",
                        help="Path to video file, or 0 for webcam (default: 0)")
    parser.add_argument("--save", default=None,
                        help="Optional: save output to this file (e.g. output.mp4)")
    parser.add_argument("--no-face", action="store_true",
                        help="Skip face recognition (faster, tests tracking only)")
    args = parser.parse_args()

    # Open video source
    source = int(args.video) if args.video == "0" else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.video}")

    # Setup video writer if saving
    writer = None
    if args.save:
        fps    = cap.get(cv2.CAP_PROP_FPS) or 30
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps, (width, height))
        print(f"[INFO] Saving output to: {args.save}")

    # Load models
    print("[INFO] Loading YOLO model...")
    yolo = YOLO("yolov8n.pt")

    face_app = None
    known_db = {}
    if not args.no_face:
        print("[INFO] Loading InsightFace model...")
        face_app = FaceAnalysis(name="buffalo_l")
        face_app.prepare(ctx_id=-1, det_size=DET_SIZE)
        known_db = load_known_faces(face_app, KNOWN_DIR)

    print(f"[INFO] Source: {args.video}")
    print("[INFO] Starting test... Press Q to stop early.\n")

    humans          = []
    next_human_id   = 1
    tracker_to_human = {}
    stats           = TrackingStats()
    frame_idx       = 0
    start_time      = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # YOLO tracking
        results = yolo.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONF_THRESH,
            classes=[0],
            verbose=False
        )
        r         = results[0]
        annotated = r.plot()

        person_boxes = {}
        current_tids = []

        if r.boxes is not None and r.boxes.id is not None:
            tids  = r.boxes.id.cpu().numpy().astype(int).tolist()
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            for tid, box in zip(tids, boxes):
                person_boxes[int(tid)] = tuple(map(float, box))
            current_tids = sorted(set(tids))

        # Face recognition
        if face_app and person_boxes and (frame_idx % RECOG_EVERY_N_FRAMES == 0):
            faces = face_app.get(frame)
            for face in faces:
                fb  = face.bbox.astype(float)
                emb = l2_normalize(face.embedding.astype(np.float32))

                assigned_tid = None
                for tid, pb in person_boxes.items():
                    if face_center_in_box(fb, pb):
                        assigned_tid = tid
                        break
                if assigned_tid is None:
                    continue

                best_human, best_dist = None, 999.0
                for h in humans:
                    d = cosine_distance(emb, h["embedding"])
                    if d < best_dist:
                        best_dist = d
                        best_human = h

                if best_human is not None and best_dist <= NEW_HUMAN_THRESHOLD:
                    tracker_to_human[assigned_tid] = best_human["human_id"]
                    best_human["embedding"] = l2_normalize(
                        0.9 * best_human["embedding"] + 0.1 * emb)
                    if best_human["name"] is None:
                        name, _ = match_known_name(best_human["embedding"], known_db)
                        if name:
                            best_human["name"] = name
                else:
                    human_id = next_human_id
                    next_human_id += 1
                    name, _ = match_known_name(emb, known_db)
                    humans.append({"human_id": human_id, "embedding": emb, "name": name})
                    tracker_to_human[assigned_tid] = human_id

        # Update stats
        stats.update(frame_idx, current_tids, tracker_to_human, humans)

        # Current humans in frame
        current_humans_in_frame = set()
        for tid in current_tids:
            hid = tracker_to_human.get(tid)
            if hid is not None:
                current_humans_in_frame.add(hid)

        # Overlays
        elapsed = int(time.time() - start_time)
        fps_live = frame_idx / elapsed if elapsed > 0 else 0

        cv2.putText(annotated,
            f"In frame: {len(current_humans_in_frame)}  |  Unique: {len(humans)}  |  FPS: {fps_live:.1f}  |  Time: {elapsed}s",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

        cv2.putText(annotated,
            f"ID switches: {len(stats.id_switch_events)}  |  Frame: {frame_idx}",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        y = 110
        for tid in current_tids[:10]:
            hid = tracker_to_human.get(tid)
            if hid is None:
                label = f"Tracker {tid} -> (unassigned)"
            else:
                h  = next((x for x in humans if x["human_id"] == hid), None)
                nm = h["name"] if (h and h["name"]) else f"Human {hid}"
                label = f"Tracker {tid} -> {nm}"
            cv2.putText(annotated, label, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y += 24

        cv2.imshow("Tracking Test (Q to quit)", annotated)

        if writer:
            writer.write(annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

        frame_idx += 1

    elapsed_total = time.time() - start_time
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    stats.print_summary(elapsed_total)


if __name__ == "__main__":
    main()
