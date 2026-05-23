"""
Visualization Utilities Module
Provides debugging visualizations and enhanced display features for detection system.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class DebugVisualizer:
    """
    Provides debugging visualizations including:
    - Face-to-person assignment overlays
    - Confidence scores
    - Face embeddings visualization
    - Assignment connections
    """
    
    # Color palettes
    COLORS = {
        "person_box": (0, 255, 0),      # Green
        "face_box": (255, 0, 0),         # Blue
        "assigned": (0, 255, 255),       # Yellow - face assigned to person
        "unassigned": (0, 165, 255),     # Orange - unassigned face
        "recognized": (0, 255, 0),       # Green - recognized person
        "unknown": (0, 0, 255),          # Red - unknown person
        "connection": (255, 255, 0),     # Cyan - assignment connection
    }
    
    @staticmethod
    def draw_person_boxes(
        frame: np.ndarray,
        person_boxes: Dict[int, Tuple[float, float, float, float]],
        tracker_to_human: Dict[int, int],
        all_humans: List[Dict],
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw person bounding boxes with tracker IDs.
        
        Args:
            frame: Input frame
            person_boxes: Dict mapping tracker ID to bbox (x1,y1,x2,y2)
            tracker_to_human: Mapping from tracker to human ID
            all_humans: List of all humans for name lookup
            thickness: Box line thickness
            
        Returns:
            Frame with drawn boxes
        """
        for tid, (x1, y1, x2, y2) in person_boxes.items():
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Determine color based on assignment status
            if tid in tracker_to_human:
                human_id = tracker_to_human[tid]
                human = next((h for h in all_humans if h["human_id"] == human_id), None)
                color = DebugVisualizer.COLORS["recognized"] if human and human.get("name") else DebugVisualizer.COLORS["person_box"]
            else:
                color = DebugVisualizer.COLORS["unassigned"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(
                frame,
                f"Tracker {tid}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )
        
        return frame
    
    @staticmethod
    def draw_face_boxes(
        frame: np.ndarray,
        faces: List,
        face_assignments: Dict,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw face bounding boxes with assignment status.
        
        Args:
            frame: Input frame
            faces: List of detected faces from InsightFace
            face_assignments: Dict mapping face index to tracker ID (or None if unassigned)
            thickness: Box line thickness
            
        Returns:
            Frame with drawn face boxes
        """
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            
            # Color based on assignment
            is_assigned = idx in face_assignments and face_assignments[idx] is not None
            color = DebugVisualizer.COLORS["assigned"] if is_assigned else DebugVisualizer.COLORS["unassigned"]
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Add confidence score
            conf = face.det_score
            cv2.putText(
                frame,
                f"Face {conf:.2f}",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1
            )
        
        return frame
    
    @staticmethod
    def draw_face_person_connections(
        frame: np.ndarray,
        faces: List,
        person_boxes: Dict[int, Tuple[float, float, float, float]],
        face_assignments: Dict,
        thickness: int = 1
    ) -> np.ndarray:
        """
        Draw lines connecting assigned faces to their person boxes.
        
        Args:
            frame: Input frame
            faces: List of detected faces
            person_boxes: Person bounding boxes
            face_assignments: Face to tracker ID mapping
            thickness: Line thickness
            
        Returns:
            Frame with drawn connections
        """
        for face_idx, face in enumerate(faces):
            if face_idx not in face_assignments:
                continue
            
            tid = face_assignments[face_idx]
            if tid is None or tid not in person_boxes:
                continue
            
            # Face center
            face_bbox = face.bbox.astype(float)
            fx = int((face_bbox[0] + face_bbox[2]) / 2)
            fy = int((face_bbox[1] + face_bbox[3]) / 2)
            
            # Person center
            px1, py1, px2, py2 = person_boxes[tid]
            px = int((px1 + px2) / 2)
            py = int((py1 + py2) / 2)
            
            # Draw connection line
            cv2.line(frame, (fx, fy), (px, py), DebugVisualizer.COLORS["connection"], thickness)
            
            # Draw circle at connection points
            cv2.circle(frame, (fx, fy), 3, DebugVisualizer.COLORS["connection"], -1)
            cv2.circle(frame, (px, py), 3, DebugVisualizer.COLORS["connection"], -1)
        
        return frame
    
    @staticmethod
    def draw_statistics_overlay(
        frame: np.ndarray,
        main_text: str,
        info_text: str,
        mapping_info: List[str],
        position_y_start: int = 40
    ) -> np.ndarray:
        """
        Draw statistics and mapping information overlay.
        
        Args:
            frame: Input frame
            main_text: Main statistics text
            info_text: Frame info text
            mapping_info: List of mapping strings
            position_y_start: Starting Y position for text
            
        Returns:
            Frame with overlay
        """
        y = position_y_start
        
        # Main statistics
        cv2.putText(
            frame,
            main_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        y += 40
        
        # Frame info
        cv2.putText(
            frame,
            info_text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 35
        
        # Mapping information
        cv2.putText(
            frame,
            "--- Tracker to Human Mapping ---",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1
        )
        y += 26
        
        for mapping_text in mapping_info[:15]:  # Limit to 15 lines
            cv2.putText(
                frame,
                mapping_text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                1
            )
            y += 26
        
        return frame
    
    @staticmethod
    def draw_distance_heatmap(
        frame: np.ndarray,
        faces: List,
        person_boxes: Dict[int, Tuple[float, float, float, float]],
        face_assignments: Dict,
        distances: Dict
    ) -> np.ndarray:
        """
        Draw distance heatmap for face-person assignments (advanced debug).
        
        Args:
            frame: Input frame
            faces: Detected faces
            person_boxes: Person bounding boxes
            face_assignments: Assignments
            distances: Dict with (face_idx, tid) -> distance
            
        Returns:
            Frame with heatmap overlay
        """
        # Create semi-transparent overlay for heatmap
        overlay = frame.copy()
        
        for face_idx, face in enumerate(faces):
            for tid in person_boxes.keys():
                key = (face_idx, tid)
                if key not in distances:
                    continue
                
                dist = distances[key]
                
                # Normalize distance to color (0 = green, 1 = red)
                normalized = min(max(dist / 1.0, 0), 1)
                color = (
                    int(255 * normalized),                    # Blue
                    int(255 * (1 - normalized)),              # Green
                    200                                        # Red constant
                )
                
                # Face center
                face_bbox = face.bbox.astype(float)
                fx = int((face_bbox[0] + face_bbox[2]) / 2)
                fy = int((face_bbox[1] + face_bbox[3]) / 2)
                
                # Person center
                px1, py1, px2, py2 = person_boxes[tid]
                px = int((px1 + px2) / 2)
                py = int((py1 + py2) / 2)
                
                # Draw faint line with distance label
                cv2.line(overlay, (fx, fy), (px, py), color, 1)
                
                # Label
                mid_x = (fx + px) // 2
                mid_y = (fy + py) // 2
                cv2.putText(
                    overlay,
                    f"{dist:.2f}",
                    (mid_x, mid_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1
                )
        
        # Blend overlay
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        return frame
    
    @staticmethod
    def draw_fps_counter(
        frame: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        Draw FPS counter on frame.
        
        Args:
            frame: Input frame
            fps: Current FPS
            position: (x, y) position for text
            
        Returns:
            Frame with FPS counter
        """
        text = f"FPS: {fps:.1f}"
        color = (0, 255, 0) if fps > 15 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        
        cv2.putText(
            frame,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )
        
        return frame


class DisplayFormatter:
    """Formats display text for various scenarios"""
    
    @staticmethod
    def format_person_label(
        tid: int,
        human_id: Optional[int],
        name: Optional[str],
        confidence: Optional[float] = None
    ) -> str:
        """Format a person label for display"""
        if human_id is None:
            return f"Tracker {tid} -> (unassigned)"
        
        label = f"Tracker {tid} -> "
        if name:
            label += f"{name} 👤"
        else:
            label += f"Human {human_id}"
        
        if confidence is not None:
            label += f" ({confidence:.2f})"
        
        return label
    
    @staticmethod
    def format_summary_table(
        current_count: int,
        recognized: int,
        unrecognized: int,
        total_seen: int,
        peak: int,
        elapsed: int
    ) -> str:
        """Format a summary table"""
        lines = [
            "╔════════════════════════════════════╗",
            "║  DETECTION SUMMARY                 ║",
            f"║  Current in frame:    {current_count:3d}         ║",
            f"║    ├─ Recognized:     {recognized:3d}         ║",
            f"║    └─ Unrecognized:   {unrecognized:3d}         ║",
            f"║  Total seen (session): {total_seen:3d}         ║",
            f"║  Peak concurrent:      {peak:3d}         ║",
            f"║  Elapsed time:        {elapsed:3d}s        ║",
            "╚════════════════════════════════════╝"
        ]
        return "\n".join(lines)
