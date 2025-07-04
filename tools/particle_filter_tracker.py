import time
import cv2
import numpy as np
import torch
from pathlib import Path
import argparse
import random
import math

# YOLOX specific imports
from yolox.data.data_augment import ValTransform
from yolox.exp import get_exp
from yolox.utils import postprocess
from shapely.geometry import Polygon, box as shapely_box

# ==============================================================================
# 0. CONSTANTS & HELPER FOR MOTION DETECTOR
# ==============================================================================

# <<< NEW: Define a placeholder for the bed rectangle. Adjust for your video.
# Format: (x1, y1, x2, y2)
BED_RECTANGLE = [(152, 4), (145, 145), (208, 151), (240, 10)]

def is_box_over_lap_polygon(box, polygon_points):
    """
    Checks if a box overlaps with a polygon using the shapely library.
    box: [x1, y1, x2, y2]
    polygon_points: list of (x, y) tuples
    """
    rect = shapely_box(box[0], box[1], box[2], box[3])
    poly = Polygon(polygon_points)
    return rect.intersects(poly)


# ==============================================================================
# 1. YOLOX PREDICTOR (UNCHANGED)
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
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        model_state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.preproc = ValTransform(legacy=False)
        print(f"YOLOXPredictor initialized. Target preproc size (H,W): {self.input_size_for_preproc}")

    def inference(self, frame):
        if frame is None or frame.size == 0: return []
        img_for_preproc, ratio_from_preproc = self.preproc(frame, None, self.input_size_for_preproc)
        if isinstance(ratio_from_preproc, tuple): ratio_to_use = ratio_from_preproc[0]
        else: ratio_to_use = float(ratio_from_preproc)
        if ratio_to_use == 0: ratio_to_use = 1.0
        img_tensor = torch.from_numpy(img_for_preproc).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            outputs = self.model(img_tensor)
            processed_outputs = postprocess(outputs, self.exp.num_classes, self.conf_threshold, self.nms_threshold, class_agnostic=False)
        detections_tensor = processed_outputs[0]
        if detections_tensor is None: return []
        detections_np = detections_tensor.cpu().numpy()
        bboxes_scaled = detections_np[:, 0:4] / ratio_to_use
        return [tuple(map(int, bbox)) for bbox in bboxes_scaled]

# =============================================================================
# 2. NEW MOTION DETECTOR CLASS
# ==============================================================================

class MotionDetector:
    """Detects moving objects using background subtraction."""
    def __init__(self, bbox_thresh=100, nms_thresh=0.1):
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.bbox_thresh = bbox_thresh
        self.nms_thresh = nms_thresh
        self.kernel = np.ones((9,9), dtype=np.uint8)
        print("MotionDetector initialized.")

    def _get_motion_mask(self, fg_mask, min_thresh=127):
        """Obtains a clean motion mask from the foreground mask."""
        _, thresh = cv2.threshold(fg_mask, min_thresh, 255, cv2.THRESH_BINARY)
        motion_mask = cv2.medianBlur(thresh, 3)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, self.kernel, iterations=1)
        return motion_mask

    def _get_contour_detections(self, mask):
        """Finds bounding boxes from contours in the mask."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            bbox = [x, y, x + w, y + h]
            if area > self.bbox_thresh and is_box_over_lap_polygon(bbox, BED_RECTANGLE):
                detections.append(bbox + [area])
        return np.array(detections) if detections else np.zeros((0, 5))

    def _non_max_suppression(self, boxes, scores):
        if len(boxes) == 0: return np.zeros((0, 4), dtype=int)
        boxes = boxes.astype(np.float32)
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return boxes[keep].astype(int)

    def detect(self, frame):
        """Main detection method for the class."""
        fg_mask = self.backSub.apply(frame, learningRate=-1) # Use auto learning rate
        motion_mask = self._get_motion_mask(fg_mask)
        detections = self._get_contour_detections(motion_mask)
        
        if len(detections) == 0:
            return [], fg_mask
            
        bboxes = detections[:, :4]
        scores = detections[:, -1]
        final_boxes = self._non_max_suppression(bboxes, scores)
        
        # Return in the same format as YOLOXPredictor: list of tuples
        return [tuple(box) for box in final_boxes], fg_mask

# ==============================================================================
# 3. PARTICLE FILTER IMPLEMENTATION (UNCHANGED)
# ==============================================================================
# (The ParticleFilter and ParticleTrackerManager classes remain exactly the same as the previous version with velocity)
def bbox_to_state(bbox):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    return np.array([cx, cy, w, h])

def state_to_bbox(state):
    cx, cy, w, h = state[:4]
    x1, y1 = int(cx - w / 2), int(cy - h / 2)
    x2, y2 = int(cx + w / 2), int(cy + h / 2)
    return (x1, y1, x2, y2)

def calculate_iou(box1, box2):
    x1_inter, y1_inter = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2_inter, y2_inter = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

class ParticleFilter:
    def __init__(self, track_id, initial_bbox, num_particles=100):
        self.id = track_id
        self.num_particles = num_particles
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        self.frames_since_update = 0
        initial_pos_state = bbox_to_state(initial_bbox)
        initial_vel_state = np.array([0, 0])
        initial_state = np.concatenate([initial_pos_state, initial_vel_state])
        self.particles = np.random.normal(initial_state, [10, 10, 5, 5, 2, 2], (self.num_particles, 6))
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.last_estimated_state = initial_state

    def predict(self, motion_noise_std=[5, 5, 2, 2, 4, 4]):
        # self.particles[:, 0] += self.particles[:, 4]
        # self.particles[:, 1] += self.particles[:, 5]
        noise = np.random.normal(0, motion_noise_std, (self.num_particles, 6))
        self.particles += noise

    def update(self, measurement_bbox):
        self.frames_since_update = 0
        iou_scores = np.zeros(self.num_particles)
        for i, particle_state in enumerate(self.particles):
            particle_bbox = state_to_bbox(particle_state)
            iou_scores[i] = calculate_iou(particle_bbox, measurement_bbox)
        self.weights = iou_scores
        if np.sum(self.weights) > 0: self.weights /= np.sum(self.weights)
        else: self.weights = np.ones(self.num_particles) / self.num_particles

    def correct_velocity(self, current_measurement_bbox):
        # current_pos_state = bbox_to_state(current_measurement_bbox)
        # observed_vx = current_pos_state[0] - self.last_estimated_state[0]
        # observed_vy = current_pos_state[1] - self.last_estimated_state[1]
        # observed_velocity = np.array([observed_vx, observed_vy])
        # velocity_noise = np.random.normal(0, [3, 3], (self.num_particles, 2))
        self.particles[:, 4:] = 0

    def resample(self):
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights.fill(1.0 / self.num_particles)

    def get_estimated_state(self):
        mean_state = np.average(self.particles, weights=self.weights, axis=0)
        self.last_estimated_state = mean_state
        return mean_state

class ParticleTrackerManager:
    def __init__(self, iou_threshold=0.2, max_unseen_frames=10):
        self.trackers = {}
        self.next_track_id = 0
        self.iou_threshold = iou_threshold
        self.max_unseen_frames = max_unseen_frames

    def update_trackers(self, detections):
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
                    if iou > best_iou: best_iou, best_det_idx = iou, i
                if best_iou > self.iou_threshold:
                    tracker.update(detections[best_det_idx])
                    tracker.correct_velocity(detections[best_det_idx])
                    tracker.resample()
                    matched_detections_indices.add(best_det_idx)
        for i, det_bbox in enumerate(detections):
            if i not in matched_detections_indices:
                self.trackers[self.next_track_id] = ParticleFilter(self.next_track_id, det_bbox)
                self.next_track_id += 1
        lost_trackers = [tid for tid, t in self.trackers.items() if t.frames_since_update > self.max_unseen_frames]
        for tid in lost_trackers:
            del self.trackers[tid]
            print(f"Removed lost tracker ID: {tid}")

    def draw_trackers(self, frame):
        for tracker in self.trackers.values():
            for particle_state in tracker.particles:
                px, py = int(particle_state[0]), int(particle_state[1])
                cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)
            estimated_state = tracker.get_estimated_state()
            bbox = state_to_bbox(estimated_state)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), tracker.color, 3)
            cx, cy, _, _, vx, vy = estimated_state
            speed = math.sqrt(vx**2 + vy**2)
            cv2.putText(frame, f"ID: {tracker.id} (Spd: {speed:.1f})", (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, tracker.color, 2)
            # if speed > 1.0:
            #     start_point = (int(cx), int(cy))
            #     end_point = (int(cx + vx * 5), int(cy + vy * 5))
            #     cv2.arrowedLine(frame, start_point, end_point, tracker.color, 2, tipLength=0.4)
        return frame

# ==============================================================================
# 4. MAIN EXECUTION (WITH TOGGLE)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Particle Filter Tracking with Toggable Detector")
    parser.add_argument("--video", type=str, default=r"F:\NTUC-P-57_202506111624_202506111636.webm", help="Path to the video file.")
    parser.add_argument("--model", type=str, default=r"C:\Users\Tairin Pairor\Documents\Github\Tarin%20Project\person-detection\models\best_ckpt 1.pth", help="Path to the YOLOX .pth model file.")
    parser.add_argument("--conf", type=float, default=0.5, help="YOLOX confidence threshold.")
    # <<< NEW: Argument to choose the detector
    parser.add_argument("--detector", type=str, default="motion", choices=["yolox", "motion"], help="Detector to use for tracking: 'yolox' or 'motion'.")
    args = parser.parse_args()

    # --- Initialize Detectors ---
    yolox_predictor = YOLOXPredictor(model_path=args.model, conf_threshold=args.conf, device="cuda" if torch.cuda.is_available() else "cpu")
    motion_detector = MotionDetector()
    
    # --- Initialize Tracker ---
    tracker_manager = ParticleTrackerManager()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 0)

        time.sleep(0.05)  # Small delay to simulate real-time processing
        
        start_time = time.time()
        
        # <<< NEW: Toggle between detectors based on argument
        if args.detector == 'yolox':
            current_detections = yolox_predictor.inference(frame)
        else: # 'motion'
            current_detections, fg_mask = motion_detector.detect(frame)
            cv2.imshow("Foreground Mask", fg_mask) # Show mask for debugging

        # The tracker manager doesn't care where the detections came from
        tracker_manager.update_trackers(current_detections)
        
        # --- Visualization ---
        display_frame = frame.copy()
        
        # Draw the raw detections from the chosen detector
        for bbox in current_detections:
            cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1) # Blue for raw detections

        # Draw the final tracked boxes and particles
        display_frame = tracker_manager.draw_trackers(display_frame)

        end_time = time.time()
        fps = 1 / (end_time - start_time) if (end_time - start_time) > 0 else 0
        cv2.putText(display_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Detector: {args.detector.upper()}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        cv2.imshow("Particle Filter Tracking", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()