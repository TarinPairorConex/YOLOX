import time
import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
import random

# YOLOX specific imports
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess

# ==============================================================================
# 1. YOLOX PREDICTOR (CORRECTED)
# ==============================================================================

class YOLOXPredictor:
    """YOLOX model for object detection"""
    
    def __init__(self, model_path, model_name="yolox-nano", conf_threshold=0.3, 
                 nms_threshold=0.45, input_size_wh=(416, 416), device="cpu"):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size_for_preproc = (input_size_wh[1], input_size_wh[0])
        self.device = torch.device(device)
        
        self.exp = get_exp(exp_file=None, exp_name=model_name)
        self.exp.num_classes = 1
        self.exp.class_names = ["person"]
        self.exp.test_size = self.input_size_for_preproc
        self.exp.test_conf = conf_threshold
        self.exp.nmsthre = nms_threshold
        
        self.model = self.exp.get_model()
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # <<< FIX 1: Added weights_only=False to match your original script
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        self.preproc = ValTransform(legacy=False)
        print(f"YOLOXPredictor initialized. Target preproc size (H,W): {self.input_size_for_preproc}")

    def inference(self, frame):
        """
        Run inference on a frame and return processed detections.
        """
        if frame is None or frame.size == 0:
            return []

        # <<< FIX 2: RESTORED THE ROBUST RATIO HANDLING FROM YOUR ORIGINAL SCRIPT
        img_for_preproc, ratio_from_preproc = self.preproc(frame, None, self.input_size_for_preproc)

        # Validate ratio - handle different types (float, tuple, numpy array)
        if not isinstance(ratio_from_preproc, (float, tuple)):
            # Calculate ratio manually if preproc output is bad
            original_h, original_w = frame.shape[:2]
            target_h, target_w = self.input_size_for_preproc
            r = min(target_h / original_h, target_w / original_w)
            ratio_to_use = r
        elif isinstance(ratio_from_preproc, tuple):
            ratio_to_use = ratio_from_preproc[0]
        else:
            ratio_to_use = float(ratio_from_preproc)
        
        if ratio_to_use == 0:
            print("ERROR: Calculated ratio is zero. Defaulting to 1.0.")
            ratio_to_use = 1.0
        
        img_tensor = torch.from_numpy(img_for_preproc).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            processed_outputs = postprocess(
                outputs, self.exp.num_classes, self.conf_threshold,
                self.nms_threshold, class_agnostic=False
            )
        
        detections_tensor = processed_outputs[0]
        if detections_tensor is None:
            return []

        detections_np = detections_tensor.cpu().numpy()
        
        # Now this division will work because ratio_to_use is a scalar
        bboxes_scaled = detections_np[:, 0:4] / ratio_to_use
        
        processed_detections = []
        for i in range(len(bboxes_scaled)):
            bbox = tuple(map(int, bboxes_scaled[i])) # (x1, y1, x2, y2)
            processed_detections.append(bbox)
            
        return processed_detections

# ==============================================================================
# 2. PARTICLE FILTER IMPLEMENTATION (UNCHANGED)
# ==============================================================================

def bbox_to_state(bbox):
    """Convert (x1, y1, x2, y2) bbox to (cx, cy, w, h) state."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2
    cy = y1 + h / 2
    return np.array([cx, cy, w, h])

def state_to_bbox(state):
    """Convert (cx, cy, w, h) state to (x1, y1, x2, y2) bbox."""
    cx, cy, w, h = state
    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)
    return (x1, y1, x2, y2)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


class ParticleFilter:
    """A particle filter for tracking a single object."""

    def __init__(self, track_id, initial_bbox, num_particles=100):
        self.id = track_id
        self.num_particles = num_particles
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.frames_since_update = 0
        initial_state = bbox_to_state(initial_bbox)
        self.particles = np.random.normal(initial_state, [10, 10, 5, 5], (self.num_particles, 4))
        self.weights = np.ones(self.num_particles) / self.num_particles

    def predict(self, motion_noise_std=[15, 15, 7, 7]):
        """Move particles according to a simple random walk motion model."""
        noise = np.random.normal(0, motion_noise_std, (self.num_particles, 4))
        self.particles += noise

    def update(self, measurement_bbox):
        """Update particle weights based on the similarity to the measurement."""
        self.frames_since_update = 0
        iou_scores = np.zeros(self.num_particles)
        for i, particle_state in enumerate(self.particles):
            particle_bbox = state_to_bbox(particle_state)
            iou_scores[i] = calculate_iou(particle_bbox, measurement_bbox)
        self.weights = iou_scores
        if np.sum(self.weights) > 0:
            self.weights /= np.sum(self.weights)
        else:
            self.weights = np.ones(self.num_particles) / self.num_particles

    def resample(self):
        """Resample particles to combat particle degeneracy."""
        indices = np.random.choice(
            np.arange(self.num_particles),
            size=self.num_particles,
            p=self.weights
        )
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def get_estimated_state(self):
        """Compute the final state estimate."""
        mean_state = np.average(self.particles, weights=self.weights, axis=0)
        return mean_state


class ParticleTrackerManager:
    """Manages all active particle filter trackers."""

    def __init__(self, iou_threshold=0.1, max_unseen_frames=10):
        self.trackers = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_unseen_frames = max_unseen_frames

    def update_trackers(self, detections):
        """Update all trackers with new detections from a frame."""
        for track_id in list(self.trackers.keys()):
            self.trackers[track_id].predict()
            self.trackers[track_id].frames_since_update += 1

        matched_detections_indices = set()
        if len(detections) > 0 and len(self.trackers) > 0:
            for track_id, tracker in self.trackers.items():
                est_bbox = state_to_bbox(tracker.get_estimated_state())
                best_iou, best_det_idx = 0, -1
                for i, det_bbox in enumerate(detections):
                    if i in matched_detections_indices: continue
                    iou = calculate_iou(est_bbox, det_bbox)
                    if iou > best_iou:
                        best_iou, best_det_idx = iou, i
                if best_iou > self.iou_threshold:
                    tracker.update(detections[best_det_idx])
                    tracker.resample()
                    matched_detections_indices.add(best_det_idx)
        
        for i, det_bbox in enumerate(detections):
            if i not in matched_detections_indices:
                new_tracker = ParticleFilter(self.next_track_id, det_bbox)
                self.trackers[self.next_track_id] = new_tracker
                self.next_track_id += 1

        lost_trackers = [tid for tid, t in self.trackers.items() if t.frames_since_update > self.max_unseen_frames]
        for tid in lost_trackers:
            del self.trackers[tid]
            print(f"Removed lost tracker ID: {tid}")

    def draw_trackers(self, frame):
        """Draw all particles and estimated bounding boxes on the frame."""
        for tracker in self.trackers.values():
            for particle_state in tracker.particles:
                px, py = int(particle_state[0]), int(particle_state[1])
                cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)
            
            estimated_state = tracker.get_estimated_state()
            bbox = state_to_bbox(estimated_state)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), tracker.color, 3)
            cv2.putText(frame, f"ID: {tracker.id}", (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracker.color, 2)
        return frame


# ==============================================================================
# 3. MAIN EXECUTION
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Particle Filter Tracking with YOLOX")
    parser.add_argument("--video", type=str, default=r"F:\NTUC-P-57_202506111624_202506111636.webm", help="Path to the video file.")
    parser.add_argument("--model", type=str, default=r"C:\Users\Tairin Pairor\Documents\Github\Tarin%20Project\person-detection\models\best_ckpt 1.pth", help="Path to the YOLOX .pth model file.")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLOX confidence threshold.")
    args = parser.parse_args()

    # Initialize YOLOX detector
    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = YOLOXPredictor(
        model_path=args.model,
        conf_threshold=args.conf,
        device=device
    )

    # Initialize Tracker Manager
    tracker_manager = ParticleTrackerManager()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        start_time = time.time()
        yolo_detections = predictor.inference(frame)
        tracker_manager.update_trackers(yolo_detections)
        display_frame = frame.copy()
        
        for bbox in yolo_detections:
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)

        display_frame = tracker_manager.draw_trackers(display_frame)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Particle Filter Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
