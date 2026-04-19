import cv2
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import time

class EgoMotionCompensator:
    def __init__(self, max_corners=200, quality_level=0.01, min_distance=10):
        self.prev_gray = None
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.lk_params = dict(
            winSize = (21, 21),
            maxLevel = 3,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def estimate_translation(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0

        # detect good feature points to track in previous frame
        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance
        )
        
        if prev_pts is None or len(prev_pts) < 4:
            self.prev_gray = gray
            return 0.0, 0.0
 
        # Track those points into the current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )
 
        # Keep only successfully tracked points
        valid_prev = prev_pts[status == 1]
        valid_curr = curr_pts[status == 1]
 
        if len(valid_prev) < 4:
            self.prev_gray = gray
            return 0.0, 0.0
 
        # Robust median translation (avoids outliers from moving objects)
        deltas = valid_curr - valid_prev
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
 
        self.prev_gray = gray
        return dx, dy

    def compensate_detections(self, detections, dx, dy):
        if detections.xyxy is None or len(detections.xyxy) == 0:
            return detections
        compensated = detections.xyxy.copy()
        compensated[:, [0, 2]] -= dx   # x1, x2
        compensated[:, [1, 3]] -= dy   # y1, y2
        detections.xyxy = compensated
        return detections

def run_pipeline(source_video_path, target_video_path):
    print("Loading YOLO11n model via SAHI...")
    
    # SAHI treats YOLOv8/11 models under the 'ultralytics' or 'yolov8' type
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path='yolo11n.pt',
        confidence_threshold=0.2, # Lower threshold for tiny objects
        device="cpu" 
    )

    # Initialize tailored drone tracker (ByteTrack)
    # We lower thresholds so it doesn't lose tracks of small humans easily
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8,
        frame_rate=30
    )
    
    # Ego-motion compensator 
    ego = EgoMotionCompensator()

    # Annotators for visualization
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50) # To show movement paths

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    
    print(f"Processing {source_video_path}...")
    print(f"Video resolution: {video_info.width}x{video_info.height}")
    
    # Instead of shrinking to 640x640, process in 512x512 patches
    slice_height = 512
    slice_width = 512
    overlap_ratio = 0.2
    
    frame_times = []
    total_frames = 0

    frame_generator = sv.get_video_frames_generator(source_video_path)

    with sv.VideoSink(target_path=target_video_path, video_info=video_info) as sink:
        for idx, frame in enumerate(frame_generator):
            t_start = time.perf_counter()

            # Ego motion estimation
            dx, dy = ego.estimate_translation(frame) 


            # 1. Run SAHI (Slicing Aided Hyper Inference)
            result = get_sliced_prediction(
                frame,
                detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                postprocess_type="NMM" #Non-maximum merging across slice boundaries
            )

            # Convert SAHI results to Supervision Detections format
            # result.object_prediction_list contains the detections
            boxes = []
            confidences = []
            class_ids = []
            
            for obj in result.object_prediction_list:
                # obj.category.id == 0 means person in COCO
                # obj.category.id == 2 means car, id == 3 means motorcycle, 
                # id == 5 means bus and id == 7 means truck
                if obj.category.id in (0, 2, 3, 5, 7):
                    bbox = obj.bbox.to_xyxy()
                    boxes.append(bbox)
                    confidences.append(obj.score.value)
                    class_ids.append(0)
            
            if len(boxes) > 0:
                detections = sv.Detections(
                    xyxy=np.array(boxes),
                    confidence=np.array(confidences),
                    class_id=np.array(class_ids)
                )
            else:
                detections = sv.Detections.empty()
            
            detections = ego.compensate_detections(detections, dx, dy)

            # 2. Update Tracker
            detections = tracker.update_with_detections(detections)

            # 3. Annotate Frame
            labels = [
                f"ID {tracker_id}"
                for tracker_id in detections.tracker_id] if detections.tracker_id is not None else []

            annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # overlay live FPS on frame
            elapsed = time.perf_counter() - t_start
            frame_times.append(elapsed)
            live_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            cv2.putText(
                annotated_frame,
                f"FPS: {live_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 2
            )
            

            sink.write_frame(annotated_frame)
            total_frames += 1
            
            if idx % 10 == 0:
                avg_so_far = 1.0 / np.mean(frame_times) if frame_times else 0
                print(f"  Frame {idx:4d} | live FPS: {live_fps:.2f} | avg FPS: {avg_so_far:.2f} | ego Δ=({dx:.1f},{dy:.1f})")

    if frame_times:
        avg_fps     = 1.0 / np.mean(frame_times)
        median_fps  = 1.0 / np.median(frame_times)
        min_fps     = 1.0 / max(frame_times)
        max_fps     = 1.0 / min(frame_times)
        print("\n" + "="*50)
        print("  PIPELINE PERFORMANCE SUMMARY")
        print("="*50)
        print(f"  Total frames processed : {total_frames}")
        print(f"  Average FPS            : {avg_fps:.2f}")
        print(f"  Median  FPS            : {median_fps:.2f}")
        print(f"  Min FPS (slowest frame): {min_fps:.2f}")
        print(f"  Max FPS (fastest frame): {max_fps:.2f}")
        print("="*50)

    print(f"Output saved to {target_video_path}")
    return {
        "avg_fps": avg_fps if frame_times else 0,
        "total_frames": total_frames
    }

if __name__ == "__main__":
    run_pipeline("drone_tracker/drone_test.avi", 
                 "drone_tracker/drone_sahi_tracking.mp4")