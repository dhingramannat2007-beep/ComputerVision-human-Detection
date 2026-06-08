import os
import glob
import time
import numpy as np
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

# ---------- CONFIG ----------
CONF_THRESH = 0.40
TRACKER = "bytetrack.yaml"
RECOG_EVERY_N_FRAMES = 5
DET_SIZE = (640, 640)

# Face thresholds (cosine distance = 1 - cosine_similarity)
NEW_HUMAN_THRESHOLD = 0.75           # match to existing human if dist <= this
KNOWN_MATCH_THRESHOLD = 0.65         # match to known name if dist <= this

KNOWN_DIR = "known_faces"            # optional folder for names

# ---------- HELPERS ----------


def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))  # assumes both normalized


def face_center_in_box(face_bbox, person_bbox) -> bool:
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def load_known_faces(face_app: FaceAnalysis, known_dir: str):
    """
    Reads known_faces/<Name>/*.jpg and builds averaged embeddings per name.
    Returns: dict name -> normalized_embedding
    """
    if not os.path.isdir(known_dir):
        return {}

    known = {}
    person_dirs = [d for d in os.listdir(
        known_dir) if os.path.isdir(os.path.join(known_dir, d))]
    if not person_dirs:
        return {}

    for person in sorted(person_dirs):
        imgs = []
        for ext in ("*.jpg", "*.jpeg", "*.png"):
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
        else:
            print(f"[WARN] No usable faces for: {person}")

    return known


def match_known_name(face_emb_norm: np.ndarray, known_db: dict):
    """
    Returns (name or None, best_dist)
    """
    if not known_db:
        return None, 999.0

    best_name = None
    best_dist = 999.0
    for name, kemb in known_db.items():
        d = cosine_distance(face_emb_norm, kemb)
        if d < best_dist:
            best_dist = d
            best_name = name

    if best_dist <= KNOWN_MATCH_THRESHOLD:
        return best_name, best_dist
    return None, best_dist

# ---------- MAIN ----------


def main():
    yolo = YOLO("yolov8n.pt")

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=DET_SIZE)  # CPU

    known_db = load_known_faces(face_app, KNOWN_DIR)
    if not known_db:
        print(
            "[INFO] No known_faces database found (or empty). Will label as Human 1, Human 2, ...")

    cap = cv2.VideoCapture("/Users/mannat/Downloads/14445443_1920_1080_30fps.mp4")
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Global human memory
    # list: {"human_id": int, "embedding": np.ndarray (normalized), "name": str or None}
    humans = []
    next_human_id = 1

    # tracker id -> human id mapping
    tracker_to_human = {}

    start_time = time.time()
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Person detection+tracking
        results = yolo.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONF_THRESH,
            classes=[0],
            verbose=False
        )
        r = results[0]
        annotated = r.plot()

        # Map tid -> bbox
        person_boxes = {}
        current_tids = []

        if r.boxes is not None and r.boxes.id is not None:
            tids = r.boxes.id.cpu().numpy().astype(int).tolist()
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            for tid, box in zip(tids, boxes):
                person_boxes[int(tid)] = tuple(map(float, box))
            current_tids = sorted(set(tids))

        # Face recognition every N frames
        if person_boxes and (frame_idx % RECOG_EVERY_N_FRAMES == 0):
            faces = face_app.get(frame)
            for face in faces:
                fb = face.bbox.astype(float)
                emb = l2_normalize(face.embedding.astype(np.float32))

                # Assign face to a tracked person (face center inside person bbox)
                assigned_tid = None
                for tid, pb in person_boxes.items():
                    if face_center_in_box(fb, pb):
                        assigned_tid = tid
                        break
                if assigned_tid is None:
                    continue

                # Match to existing humans
                best_human = None
                best_dist = 999.0
                for h in humans:
                    d = cosine_distance(emb, h["embedding"])
                    if d < best_dist:
                        best_dist = d
                        best_human = h

                if best_human is not None and best_dist <= NEW_HUMAN_THRESHOLD:
                    # Existing human
                    human_id = best_human["human_id"]
                    tracker_to_human[assigned_tid] = human_id

                    # Update embedding slowly (stabilizes)
                    best_human["embedding"] = l2_normalize(
                        0.9 * best_human["embedding"] + 0.1 * emb)

                    # Try naming if unknown
                    if best_human["name"] is None:
                        name, _ = match_known_name(
                            best_human["embedding"], known_db)
                        if name:
                            best_human["name"] = name
                else:
                    # New human
                    human_id = next_human_id
                    next_human_id += 1

                    name, _ = match_known_name(emb, known_db)
                    humans.append({
                        "human_id": human_id,
                        "embedding": emb,
                        "name": name
                    })
                    tracker_to_human[assigned_tid] = human_id

        # -------- NEW: Current humans in frame --------
        current_humans_in_frame = set()
        for tid in current_tids:
            hid = tracker_to_human.get(tid)
            if hid is not None:
                current_humans_in_frame.add(hid)

        # Overlays
        elapsed = int(time.time() - start_time)
        cv2.putText(
            annotated,
            f"Current in frame: {len(current_humans_in_frame)}  |  Unique seen: {len(humans)}  |  Time: {elapsed}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        cv2.putText(
            annotated,
            f"Current tracker IDs: {current_tids}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Show mapping + names
        y = 120
        for tid in current_tids[:10]:
            hid = tracker_to_human.get(tid, None)
            if hid is None:
                label = f"Tracker {tid} -> (unassigned)"
            else:
                h = next((x for x in humans if x["human_id"] == hid), None)
                nm = h["name"] if (h and h["name"]) else f"Human {hid}"
                label = f"Tracker {tid} -> {nm}"
            cv2.putText(
                annotated, label, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2
            )
            y += 26

        cv2.imshow("People Counter + Face Identity (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n[RESULT]")
    print(f"Unique humans seen: {len(humans)}")
    print()

    recognized = [h for h in humans if h["name"] is not None]
    unrecognized = [h for h in humans if h["name"] is None]

    if recognized:
        print("=== Recognized People ===")
    for h in recognized:
        print(
            f"  Hi {h['name']}! 👋  (appeared as tracker Human {h['human_id']})")

    if unrecognized:
        print("\n=== Unrecognized People ===")
    for h in unrecognized:
        print(f"  Unknown person (Human {h['human_id']})")

    if not recognized and not unrecognized:
        print("  No one was detected.")


if __name__ == "__main__":
    main()
