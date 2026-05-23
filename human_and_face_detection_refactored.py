"""
Human and Face Detection with Improved Counting
Complete refactored version with proper statistics tracking, visualization debugging,
and persistence.

Key Improvements:
- Proper human counting (current vs total)
- Face-person assignment debugging
- Stale tracker cleanup
- Session reporting
- Better naming and structure
"""

import os
import glob
import time
import numpy as np
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from detection_stats import HumanCounter, TrackerCleanup
from visualization_utils import DebugVisualizer, DisplayFormatter

# ---------- CONFIG ----------
CONF_THRESH = 0.40
TRACKER = "bytetrack.yaml"
RECOG_EVERY_N_FRAMES = 5
DET_SIZE = (640, 640)

# Face thresholds (cosine distance = 1 - cosine_similarity)
NEW_HUMAN_THRESHOLD = 0.75           # match to existing human if dist <= this
KNOWN_MATCH_THRESHOLD = 0.65         # match to known name if dist <= this

KNOWN_DIR = "known_faces"            # optional folder for names

# Debug flags
DEBUG_VISUALIZE = True               # Show assignment connections and heatmaps
DEBUG_SHOW_DISTANCE_HEATMAP = False  # Show embedding distance heatmap
CLEANUP_STALE_TRACKERS = True        # Clean up tracker mappings
STALE_TRACKER_FRAMES = 30            # Frames before tracker considered stale

# ---------- HELPERS ----------


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length"""
    return v / (np.linalg.norm(v) + 1e-12)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (assumes both normalized)"""
    return float(1.0 - np.dot(a, b))


def face_center_in_box(face_bbox, person_bbox) -> bool:
    """Check if face center is inside person bounding box"""
    fx1, fy1, fx2, fy2 = face_bbox
    px1, py1, px2, py2 = person_bbox
    cx = (fx1 + fx2) / 2.0
    cy = (fy1 + fy2) / 2.0
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def find_closest_person(face_bbox, person_boxes):
    """
    Find closest person box if face overlaps multiple persons.
    
    Returns:
        Tuple of (closest_tid, distance) or (None, inf)
    """
    fx_center = (face_bbox[0] + face_bbox[2]) / 2
    fy_center = (face_bbox[1] + face_bbox[3]) / 2
    
    closest_tid = None
    min_distance = float('inf')
    
    for tid, pb in person_boxes.items():
        if face_center_in_box(face_bbox, pb):
            px_center = (pb[0] + pb[2]) / 2
            py_center = (pb[1] + pb[3]) / 2
            dist = ((fx_center - px_center)**2 + (fy_center - py_center)**2)**0.5
            
            if dist < min_distance:
                min_distance = dist
                closest_tid = tid
    
    return closest_tid, min_distance


def load_known_faces(face_app: FaceAnalysis, known_dir: str):
    """
    Load known faces from directory structure: known_faces/<Name>/*.jpg
    
    Returns:
        Dict mapping name -> normalized_embedding
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
    Match face embedding to known database.
    
    Returns:
        Tuple of (name or None, best_distance)
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
    print("[INFO] Initializing models...")
    yolo = YOLO("yolov8n.pt")

    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(ctx_id=-1, det_size=DET_SIZE)  # CPU

    known_db = load_known_faces(face_app, KNOWN_DIR)
    if not known_db:
        print(
            "[INFO] No known_faces database found (or empty). "
            "Will label as Human 1, Human 2, ...")

    print("[INFO] Opening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # Initialize statistics and cleanup managers
    counter = HumanCounter(log_dir="detection_logs")
    cleanup_manager = TrackerCleanup(stale_threshold=STALE_TRACKER_FRAMES)

    # Global human memory
    # List of: {"human_id": int, "embedding": np.ndarray (normalized), "name": str or None}
    humans = []
    next_human_id = 1

    # Tracker ID -> Human ID mapping
    tracker_to_human = {}

    # FPS calculation
    start_time = time.time()
    frame_idx = 0
    fps_times = []

    print("[INFO] Starting detection loop (Q to quit)...")
    print("-" * 60)

    while True:
        frame_time = time.time()
        ok, frame = cap.read()
        if not ok:
            break

        # ===== PERSON DETECTION & TRACKING =====
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

        # Extract tracked persons
        person_boxes = {}
        current_tids = []

        if r.boxes is not None and r.boxes.id is not None:
            tids = r.boxes.id.cpu().numpy().astype(int).tolist()
            boxes = r.boxes.xyxy.cpu().numpy().tolist()
            for tid, box in zip(tids, boxes):
                person_boxes[int(tid)] = tuple(map(float, box))
            current_tids = sorted(set(tids))

        # ===== STALE TRACKER CLEANUP =====
        if CLEANUP_STALE_TRACKERS:
            stale_tids = cleanup_manager.update_active_trackers(current_tids, frame_idx)
            tracker_to_human = cleanup_manager.cleanup_dead_mappings(tracker_to_human, stale_tids)

        # ===== FACE RECOGNITION (every N frames) =====
        face_assignments = {}  # face_idx -> assigned_tid (for visualization)
        
        if person_boxes and (frame_idx % RECOG_EVERY_N_FRAMES == 0):
            faces = face_app.get(frame)
            
            for face_idx, face in enumerate(faces):
                counter.add_detected_face()
                
                fb = face.bbox.astype(float)
                emb = l2_normalize(face.embedding.astype(np.float32))

                # Assign face to closest tracked person
                assigned_tid, _ = find_closest_person(fb, person_boxes)
                
                if assigned_tid is None:
                    face_assignments[face_idx] = None
                    continue
                
                face_assignments[face_idx] = assigned_tid

                # Match to existing humans by embedding
                best_human = None
                best_dist = 999.0
                for h in humans:
                    d = cosine_distance(emb, h["embedding"])
                    if d < best_dist:
                        best_dist = d
                        best_human = h

                if best_human is not None and best_dist <= NEW_HUMAN_THRESHOLD:
                    # ===== EXISTING HUMAN =====
                    human_id = best_human["human_id"]
                    tracker_to_human[assigned_tid] = human_id

                    # Update embedding (exponential smoothing)
                    best_human["embedding"] = l2_normalize(
                        0.9 * best_human["embedding"] + 0.1 * emb)

                    # Try naming if unknown
                    if best_human["name"] is None:
                        name, _ = match_known_name(best_human["embedding"], known_db)
                        if name:
                            best_human["name"] = name
                else:
                    # ===== NEW HUMAN =====
                    human_id = next_human_id
                    next_human_id += 1

                    name, _ = match_known_name(emb, known_db)
                    new_human = {
                        "human_id": human_id,
                        "embedding": emb,
                        "name": name
                    }
                    humans.append(new_human)
                    tracker_to_human[assigned_tid] = human_id
                    counter.record_new_human(new_human)

        # ===== UPDATE STATISTICS =====
        current_count, recognized_count, unrecognized_count = counter.update_frame_detections(
            current_tids,
            tracker_to_human,
            humans
        )

        # ===== VISUALIZATION =====
        elapsed = int(time.time() - start_time)

        # Main statistics text
        main_stats_text = counter.get_display_text(elapsed)
        
        # Frame info text
        frame_info_text = counter.get_frame_info_text(current_tids)

        # Generate mapping information for display
        mapping_info = []
        for tid in current_tids[:10]:
            hid = tracker_to_human.get(tid, None)
            if hid is None:
                label = DisplayFormatter.format_person_label(tid, None, None)
            else:
                h = next((x for x in humans if x["human_id"] == hid), None)
                name = h["name"] if (h and h["name"]) else None
                label = DisplayFormatter.format_person_label(tid, hid, name)
            mapping_info.append(label)

        # Draw statistics overlay
        annotated = DebugVisualizer.draw_statistics_overlay(
            annotated,
            main_stats_text,
            frame_info_text,
            mapping_info
        )

        # Debug visualization
        if DEBUG_VISUALIZE and person_boxes:
            annotated = DebugVisualizer.draw_person_boxes(
                annotated,
                person_boxes,
                tracker_to_human,
                humans,
                thickness=2
            )
            
            if frame_idx % RECOG_EVERY_N_FRAMES == 0:
                faces = face_app.get(frame)
                annotated = DebugVisualizer.draw_face_boxes(
                    annotated,
                    faces,
                    face_assignments,
                    thickness=2
                )
                
                # Draw connections
                annotated = DebugVisualizer.draw_face_person_connections(
                    annotated,
                    faces,
                    person_boxes,
                    face_assignments,
                    thickness=1
                )

        # FPS counter
        fps_times.append(time.time() - frame_time)
        if len(fps_times) > 30:
            fps_times.pop(0)
        avg_fps = 1.0 / np.mean(fps_times) if fps_times else 0
        annotated = DebugVisualizer.draw_fps_counter(annotated, avg_fps)

        # Display
        cv2.imshow("People Counter + Face Identity (Q to quit)", annotated)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # ===== FINAL REPORT =====
    print("\n" + "="*60)
    counter.print_session_summary()
    
    recognized = [h for h in humans if h["name"] is not None]
    unrecognized = [h for h in humans if h["name"] is None]

    if recognized:
        print("=== Recognized People ===")
        for h in recognized:
            print(f"  ✓ {h['name']} (Human {h['human_id']})")

    if unrecognized:
        print("\n=== Unrecognized People ===")
        for h in unrecognized:
            print(f"  ? Human {h['human_id']}")

    if not recognized and not unrecognized:
        print("  No one was detected.")
    
    print("="*60)
    
    # Save report
    report_path = counter.save_session_report()
    print(f"\n[INFO] Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
