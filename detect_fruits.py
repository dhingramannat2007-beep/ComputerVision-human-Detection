import cv2
import time
from inference import get_model

# ---------- CONFIG ----------
# Step 1: Sign up free at https://roboflow.com
# Step 2: Go to https://app.roboflow.com/settings/api and copy your API key
# Step 3: Paste it below

ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE"

# This model detects 20+ fruits:
# Apple, Avocado, Banana, Cherry, Grape, Guava, Kiwi, Lemon, Lime,
# Mango, Orange, Papaya, Peach, Pear, Pineapple, Plum,
# Pomegranate, Strawberry, Watermelon and more!
MODEL_ID = "fruit-detection/1"

CONF_THRESH = 0.40

# Assign a different color to each fruit automatically
COLORS = [
    (255,   0,   0),  # Blue
    (  0, 255,   0),  # Green
    (  0,   0, 255),  # Red
    (255, 165,   0),  # Orange
    (255,   0, 255),  # Magenta
    (  0, 255, 255),  # Cyan
    (128,   0, 128),  # Purple
    (255,  20, 147),  # Pink
    (  0, 128,   0),  # Dark Green
    (255, 140,   0),  # Dark Orange
    ( 70, 130, 180),  # Steel Blue
    (220,  20,  60),  # Crimson
]

def get_color(class_name: str) -> tuple:
    """Give each fruit a consistent color based on its name."""
    return COLORS[hash(class_name) % len(COLORS)]


# ---------- MAIN ----------

def main():
    if ROBOFLOW_API_KEY == "YOUR_API_KEY_HERE":
        print("[ERROR] Please add your Roboflow API key in the code!")
        print("        Sign up free at https://roboflow.com")
        print("        Then go to https://app.roboflow.com/settings/api")
        return

    print("[INFO] Loading fruit detection model...")
    print("[INFO] (First run will download the model — may take a moment)")
    model = get_model(model_id=MODEL_ID, api_key=ROBOFLOW_API_KEY)
    print("[INFO] Model loaded! Opening webcam...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Make sure it is connected.")

    print("[INFO] Webcam ready. Hold a fruit in front of the camera!")
    print("[INFO] Press Q to quit.\n")

    start_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[ERROR] Failed to read from webcam.")
            break

        # Run fruit detection
        results = model.infer(frame, confidence=CONF_THRESH)
        predictions = results[0].predictions if results else []

        annotated = frame.copy()
        detected_names = []

        for pred in predictions:
            # Roboflow gives center x,y + width,height — convert to corners
            cx, cy = int(pred.x), int(pred.y)
            w,  h  = int(pred.width), int(pred.height)
            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2

            label = pred.class_name.capitalize()
            conf  = pred.confidence
            color = get_color(label)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Draw label background box
            text = f"{label}  {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)

            # Draw label text
            cv2.putText(annotated, text, (x1 + 4, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            detected_names.append(label)

        # ---------- Top overlay ----------
        elapsed = int(time.time() - start_time)

        cv2.putText(
            annotated,
            f"Fruits in frame: {len(detected_names)}  |  Time: {elapsed}s",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        if detected_names:
            cv2.putText(
                annotated,
                f"Detected: {', '.join(set(detected_names))}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2
            )
        else:
            cv2.putText(
                annotated,
                "No fruit detected — hold a fruit in front of the camera!",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2
            )

        cv2.imshow("Fruit Detector — 20+ Fruits (Q to quit)", annotated)

        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\n[DONE] Fruit detector closed.")


if __name__ == "__main__":
    main()
