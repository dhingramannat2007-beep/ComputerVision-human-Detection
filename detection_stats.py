"""
Detection Statistics Module
Handles tracking, counting, and persistence of human detection metrics.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum


class PersonStatus(Enum):
    """Status of a detected person"""
    CURRENT = "current"  # Currently visible in frame
    LEFT = "left"  # Was in frame but left
    UNKNOWN = "unknown"  # Not yet recognized


@dataclass
class DetectionStatistics:
    """Comprehensive detection statistics"""
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    total_unique_humans: int = 0
    current_in_frame: int = 0
    recognized_count: int = 0
    unrecognized_count: int = 0
    peak_concurrent: int = 0
    total_frames_processed: int = 0
    faces_detected: int = 0
    
    # Detailed history
    human_history: List[Dict] = field(default_factory=list)
    frame_history: List[Dict] = field(default_factory=list)


class HumanCounter:
    """
    Manages human detection counting, statistics, and persistence.
    Properly distinguishes between:
    - Total unique humans seen (session lifetime)
    - Current humans in frame (real-time)
    - Recognized vs unrecognized individuals
    """
    
    def __init__(self, log_dir: str = "detection_logs"):
        """
        Initialize the counter.
        
        Args:
            log_dir: Directory to save detection logs
        """
        self.log_dir = log_dir
        self.stats = DetectionStatistics()
        self.current_tracked_persons: Set[int] = set()  # Current tracker IDs in frame
        self.current_assigned_humans: Set[int] = set()  # Current human IDs in frame
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
    
    def update_frame_detections(
        self,
        tracked_person_ids: List[int],
        tracker_to_human_map: Dict[int, int],
        all_humans: List[Dict]
    ) -> Tuple[int, int, int]:
        """
        Update statistics for current frame.
        
        Args:
            tracked_person_ids: List of tracker IDs currently in frame
            tracker_to_human_map: Mapping from tracker ID to human ID
            all_humans: List of all detected humans (session lifetime)
            
        Returns:
            Tuple of (current_count, recognized_count, unrecognized_count)
        """
        self.current_tracked_persons = set(tracked_person_ids)
        
        # Calculate current humans in frame
        self.current_assigned_humans = set()
        for tid in tracked_person_ids:
            if tid in tracker_to_human_map:
                human_id = tracker_to_human_map[tid]
                self.current_assigned_humans.add(human_id)
        
        # Include unassigned persons (not yet face-recognized but detected)
        unassigned_persons_count = len(tracked_person_ids) - len(self.current_assigned_humans)
        
        current_count = len(self.current_assigned_humans) + unassigned_persons_count
        
        # Count recognized vs unrecognized among current
        recognized_current = 0
        for human_id in self.current_assigned_humans:
            human = self._find_human_by_id(human_id, all_humans)
            if human and human.get("name"):
                recognized_current += 1
        
        unrecognized_current = current_count - recognized_current
        
        # Update statistics
        self.stats.current_in_frame = current_count
        self.stats.total_unique_humans = len(all_humans)
        self.stats.recognized_count = recognized_current
        self.stats.unrecognized_count = unrecognized_current
        
        # Track peak concurrent
        if current_count > self.stats.peak_concurrent:
            self.stats.peak_concurrent = current_count
        
        self.stats.total_frames_processed += 1
        
        # Record frame history
        self.stats.frame_history.append({
            "frame": self.stats.total_frames_processed,
            "current_count": current_count,
            "recognized": recognized_current,
            "unrecognized": unrecognized_current,
            "timestamp": datetime.now().isoformat()
        })
        
        return current_count, recognized_current, unrecognized_current
    
    def add_detected_face(self):
        """Increment face detection counter"""
        self.stats.faces_detected += 1
    
    def record_new_human(self, human: Dict):
        """Record a newly detected human"""
        self.stats.human_history.append({
            "human_id": human["human_id"],
            "name": human.get("name", "Unknown"),
            "detection_time": datetime.now().isoformat()
        })
    
    def get_statistics(self) -> DetectionStatistics:
        """Get current statistics"""
        return self.stats
    
    def get_display_text(self, elapsed_seconds: int) -> str:
        """
        Generate display text for video overlay.
        
        Args:
            elapsed_seconds: Seconds elapsed since session start
            
        Returns:
            Formatted text for display
        """
        return (
            f"👤 Current: {self.stats.current_in_frame} "
            f"(✓ {self.stats.recognized_count} | ? {self.stats.unrecognized_count})  |  "
            f"Total seen: {self.stats.total_unique_humans}  |  "
            f"Peak: {self.stats.peak_concurrent}  |  "
            f"Time: {elapsed_seconds}s"
        )
    
    def get_frame_info_text(self, tracked_ids: List[int]) -> str:
        """
        Generate frame info text for video overlay.
        
        Args:
            tracked_ids: Current tracker IDs
            
        Returns:
            Formatted text showing tracked IDs
        """
        if not tracked_ids:
            return "No persons detected"
        return f"Tracked IDs: {tracked_ids} ({len(tracked_ids)} person(s))"
    
    def save_session_report(self, output_file: Optional[str] = None) -> str:
        """
        Save detailed session report to JSON file.
        
        Args:
            output_file: Path to save report. If None, generates timestamped filename.
            
        Returns:
            Path to saved report file
        """
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(self.log_dir, f"detection_report_{timestamp}.json")
        
        report = {
            "summary": asdict(self.stats),
            "session_duration_frames": self.stats.total_frames_processed,
            "average_per_frame": (
                self.stats.faces_detected / max(self.stats.total_frames_processed, 1)
            )
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[INFO] Session report saved to: {output_file}")
        return output_file
    
    def print_session_summary(self):
        """Print comprehensive session summary to console"""
        print("\n" + "="*60)
        print("SESSION SUMMARY")
        print("="*60)
        print(f"Total unique humans seen: {self.stats.total_unique_humans}")
        print(f"Peak concurrent humans: {self.stats.peak_concurrent}")
        print(f"Total frames processed: {self.stats.total_frames_processed}")
        print(f"Total faces detected: {self.stats.faces_detected}")
        
        if self.stats.total_frames_processed > 0:
            print(f"Avg faces per frame: {self.stats.faces_detected / self.stats.total_frames_processed:.2f}")
        
        if self.stats.human_history:
            print("\n" + "-"*60)
            print("DETECTION TIMELINE")
            print("-"*60)
            for i, h in enumerate(self.stats.human_history[:20], 1):  # Show first 20
                print(f"  {i}. Human {h['human_id']}: {h['name']} "
                      f"(detected at {h['detection_time']})")
            
            if len(self.stats.human_history) > 20:
                print(f"  ... and {len(self.stats.human_history) - 20} more")
        
        print("="*60 + "\n")
    
    def _find_human_by_id(self, human_id: int, all_humans: List[Dict]) -> Optional[Dict]:
        """Helper to find human by ID"""
        for h in all_humans:
            if h.get("human_id") == human_id:
                return h
        return None


class TrackerCleanup:
    """Manages cleanup of stale tracker IDs and dead mappings"""
    
    def __init__(self, stale_threshold: int = 30):
        """
        Initialize cleanup manager.
        
        Args:
            stale_threshold: Frames to keep mapping before cleanup
        """
        self.stale_threshold = stale_threshold
        self.tracker_last_seen: Dict[int, int] = {}  # tid -> frame_idx
    
    def update_active_trackers(
        self,
        current_tids: List[int],
        frame_idx: int
    ) -> Set[int]:
        """
        Update tracker activity and return stale IDs to remove.
        
        Args:
            current_tids: Currently visible tracker IDs
            frame_idx: Current frame index
            
        Returns:
            Set of stale tracker IDs to remove from mappings
        """
        # Mark current trackers as active
        for tid in current_tids:
            self.tracker_last_seen[tid] = frame_idx
        
        # Find stale trackers
        stale = set()
        for tid, last_frame in list(self.tracker_last_seen.items()):
            if frame_idx - last_frame > self.stale_threshold:
                stale.add(tid)
                del self.tracker_last_seen[tid]
        
        return stale
    
    def cleanup_dead_mappings(
        self,
        tracker_to_human: Dict[int, int],
        stale_tids: Set[int]
    ) -> Dict[int, int]:
        """
        Clean dead tracker mappings.
        
        Args:
            tracker_to_human: Current mapping dictionary
            stale_tids: IDs to remove
            
        Returns:
            Cleaned mapping dictionary
        """
        for tid in stale_tids:
            if tid in tracker_to_human:
                del tracker_to_human[tid]
        
        return tracker_to_human
