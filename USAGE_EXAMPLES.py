"""
Usage Examples - Human and Face Detection with Improved Counting

This file demonstrates various ways to use the refactored detection system.
"""

# ============================================================================
# EXAMPLE 1: Basic Usage with All Defaults
# ============================================================================

from human_and_face_detection_refactored import main

# Run with default configuration
if __name__ == "__main__":
    main()


# ============================================================================
# EXAMPLE 2: Custom Configuration for High-Accuracy Recognition
# ============================================================================

"""
File: custom_high_accuracy.py

For scenarios where accuracy is more important than speed
(e.g., security, VIP recognition)
"""

# Modify these in human_and_face_detection_refactored.py before running:

"""
# CONFIG SECTION
CONF_THRESH = 0.60                   # Higher confidence threshold
RECOG_EVERY_N_FRAMES = 2             # More frequent face recognition
NEW_HUMAN_THRESHOLD = 0.60           # Stricter human matching
KNOWN_MATCH_THRESHOLD = 0.50         # Stricter known identity matching
DEBUG_VISUALIZE = True               # Enable visualization for debugging
"""


# ============================================================================
# EXAMPLE 3: High-Speed Counting (Real-time Footfall Counter)
# ============================================================================

"""
File: footfall_counter.py

For scenarios where only count matters, not recognition
(e.g., retail footfall, visitor counting)
"""

"""
# CONFIG SECTION
CONF_THRESH = 0.30                   # Lower threshold for detection
RECOG_EVERY_N_FRAMES = 15            # Infrequent recognition (saves CPU)
NEW_HUMAN_THRESHOLD = 0.85           # Lenient matching (same person)
DEBUG_VISUALIZE = False              # Disable visualization (saves GPU)
CLEANUP_STALE_TRACKERS = True        # Clean memory regularly
STALE_TRACKER_FRAMES = 20            # More aggressive cleanup
"""


# ============================================================================
# EXAMPLE 4: Using HumanCounter Standalone for Custom Integration
# ============================================================================

from detection_stats import HumanCounter
import cv2
import numpy as np

def custom_detection_pipeline():
    """Example: Use HumanCounter in your own custom pipeline"""
    
    counter = HumanCounter(log_dir="my_detection_logs")
    humans = []
    tracker_to_human = {}
    
    # Your detection and tracking code here...
    
    # Update statistics for each frame
    current_tids = [1, 2, 3]  # Your tracked IDs
    current_count, recognized, unrecognized = counter.update_frame_detections(
        current_tids,
        tracker_to_human,
        humans
    )
    
    # Get display text
    display_text = counter.get_display_text(elapsed_seconds=123)
    print(display_text)
    # Output: 👤 Current: 3 (✓ 1 | ? 2) | Total seen: 5 | Peak: 4 | Time: 123s
    
    # At end of session
    counter.save_session_report()
    counter.print_session_summary()


# ============================================================================
# EXAMPLE 5: Using DebugVisualizer for Custom Visualization
# ============================================================================

from visualization_utils import DebugVisualizer, DisplayFormatter

def visualization_example():
    """Example: Use visualization utilities independently"""
    
    frame = cv2.imread("sample_frame.jpg")
    
    # Example data
    person_boxes = {
        1: (100, 100, 300, 500),  # tid: (x1, y1, x2, y2)
        2: (400, 150, 600, 480),
    }
    tracker_to_human = {1: 101, 2: 102}
    humans = [
        {"human_id": 101, "name": "Alice", "embedding": None},
        {"human_id": 102, "name": None, "embedding": None},
    ]
    
    # Draw person boxes
    frame = DebugVisualizer.draw_person_boxes(
        frame,
        person_boxes,
        tracker_to_human,
        humans,
        thickness=2
    )
    
    # Draw statistics overlay
    frame = DebugVisualizer.draw_statistics_overlay(
        frame,
        main_text="👤 Current: 2 (✓ 1 | ? 1) | Total: 5",
        info_text="Tracked IDs: [1, 2]",
        mapping_info=[
            "Tracker 1 -> Alice",
            "Tracker 2 -> Human 102"
        ]
    )
    
    # Draw FPS
    frame = DebugVisualizer.draw_fps_counter(frame, fps=28.5)
    
    cv2.imshow("Debug Visualization", frame)
    cv2.waitKey(0)


# ============================================================================
# EXAMPLE 6: Analyzing Saved Reports
# ============================================================================

import json
from datetime import datetime

def analyze_detection_report(report_path):
    """Example: Analyze a saved detection report"""
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    summary = report['summary']
    
    # Print basic statistics
    print("=== Session Report Analysis ===")
    print(f"Session Start: {summary['session_start']}")
    print(f"Total Frames: {summary['total_frames_processed']}")
    print(f"Unique Humans: {summary['total_unique_humans']}")
    print(f"Peak Concurrent: {summary['peak_concurrent']}")
    print(f"Faces Detected: {summary['faces_detected']}")
    
    # Calculate statistics
    avg_faces_per_frame = summary['faces_detected'] / max(summary['total_frames_processed'], 1)
    print(f"\nAverage faces per frame: {avg_faces_per_frame:.2f}")
    
    # Print timeline
    print("\n=== Detection Timeline ===")
    for entry in summary['human_history'][:10]:
        print(f"  Human {entry['human_id']}: {entry['name']} "
              f"(detected at {entry['detection_time']})")
    
    # Find peak activity time
    frame_history = summary['frame_history']
    if frame_history:
        max_concurrent_frame = max(frame_history, key=lambda x: x['current_count'])
        print(f"\nPeak activity at frame {max_concurrent_frame['frame']}: "
              f"{max_concurrent_frame['current_count']} people")


# ============================================================================
# EXAMPLE 7: Multi-session Analysis
# ============================================================================

import os
from pathlib import Path

def compare_multiple_sessions():
    """Example: Compare statistics across multiple sessions"""
    
    log_dir = "detection_logs"
    
    if not os.path.exists(log_dir):
        print(f"No logs found in {log_dir}")
        return
    
    sessions = []
    
    for report_file in sorted(os.listdir(log_dir)):
        if report_file.endswith('.json'):
            with open(os.path.join(log_dir, report_file), 'r') as f:
                report = json.load(f)
                sessions.append({
                    'file': report_file,
                    'data': report['summary']
                })
    
    if not sessions:
        print("No session reports found")
        return
    
    print("=== Multi-Session Comparison ===")
    print(f"{'Session':<30} {'People':<10} {'Peak':<10} {'Frames':<10}")
    print("-" * 60)
    
    total_people = 0
    total_peak = 0
    
    for session in sessions:
        data = session['data']
        print(f"{session['file']:<30} "
              f"{data['total_unique_humans']:<10} "
              f"{data['peak_concurrent']:<10} "
              f"{data['total_frames_processed']:<10}")
        total_people += data['total_unique_humans']
        total_peak = max(total_peak, data['peak_concurrent'])
    
    print("-" * 60)
    print(f"{'TOTAL':<30} {total_people:<10} {total_peak:<10}")


# ============================================================================
# EXAMPLE 8: Tracking Specific Individual
# ============================================================================

from detection_stats import HumanCounter

def track_individual_throughout_session(target_name):
    """Example: Track when a specific individual appears"""
    
    log_dir = "detection_logs"
    
    for report_file in os.listdir(log_dir):
        if not report_file.endswith('.json'):
            continue
        
        with open(os.path.join(log_dir, report_file), 'r') as f:
            report = json.load(f)
        
        history = report['summary']['human_history']
        
        for entry in history:
            if entry['name'] == target_name:
                print(f"✓ {target_name} detected in session {report_file}")
                print(f"  Detection time: {entry['detection_time']}")
                print(f"  Human ID: {entry['human_id']}")


# ============================================================================
# EXAMPLE 9: Real-time Statistics Publishing (to file or network)
# ============================================================================

import json
from datetime import datetime
import threading
import time

def publish_statistics_realtime(counter, interval_seconds=30):
    """
    Example: Continuously publish statistics to a JSON file
    (useful for dashboard/monitoring integration)
    """
    
    def publish_loop():
        while True:
            stats = counter.get_statistics()
            
            # Convert to dict (dataclass)
            from dataclasses import asdict
            stats_dict = asdict(stats)
            
            # Add timestamp
            stats_dict['timestamp'] = datetime.now().isoformat()
            
            # Write to file
            with open("current_stats.json", 'w') as f:
                json.dump(stats_dict, f, indent=2)
            
            time.sleep(interval_seconds)
    
    # Run in background thread
    thread = threading.Thread(target=publish_loop, daemon=True)
    thread.start()


# ============================================================================
# EXAMPLE 10: Alert System for Specific Individuals
# ============================================================================

class PersonAlertSystem:
    """Alert when VIP or flagged individuals are detected"""
    
    def __init__(self, alert_list):
        """
        Args:
            alert_list: List of names to alert on
                e.g., ['CEO', 'Security Guard', 'Unauthorized Person']
        """
        self.alert_list = alert_list
        self.alerted = set()  # Track who we've already alerted for
    
    def check_frame(self, humans):
        """Check current frame for alert individuals"""
        
        for human in humans:
            if human['name'] in self.alert_list:
                human_id = human['human_id']
                
                if human_id not in self.alerted:
                    self.alert(human['name'], human_id)
                    self.alerted.add(human_id)
    
    def alert(self, name, human_id):
        """Trigger alert for detected individual"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"🚨 ALERT: {name} detected (ID: {human_id}) at {timestamp}")
        
        # Could also:
        # - Send email
        # - Trigger sound
        # - Log to database
        # - Send webhook


# Usage in main loop:
"""
alert_system = PersonAlertSystem(['CEO', 'Security Breach', 'VIP Guest'])

# In main detection loop:
if frame_idx % 5 == 0:
    alert_system.check_frame(humans)
"""


if __name__ == "__main__":
    print("See individual examples above for usage")
    print("\nRun specific examples:")
    print("  python example_script.py  # for Example 1")
    print("  python -c 'from usage_examples import analyze_detection_report; ...'")
