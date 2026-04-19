import cv2
import time
import numpy as np
from ultralytics import YOLO

def run_baseline(video_path, output_path="baseline_output.mp4"):
    # Load the YOLO11 nano model 
    model = YOLO("yolo11n.pt")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties for saving the output
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing baseline detection on {video_path}...")
    

    frame_times = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.perf_counter()
            
        # Run inference (classes=0 restricts detection to only 'person')
        results = model.predict(frame, classes=[0], verbose=False)
        
        # Plot detections on the frame
        annotated_frame = results[0].plot()
        
        elapsed = time.perf_counter() - t0
        frame_times.append(elapsed)
 
        # Overlay live FPS on frame so it appears in saved video too
        live_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        cv2.putText(
            annotated,
            f"[Baseline] FPS: {live_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 0, 255), 2
        )

        out.write(annotated_frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            avg = 1.0 / np.mean(frame_times)
            print(f"  Frame {frame_count:4d} | live FPS: {live_fps:.2f} | avg FPS: {avg:.2f}")

    cap.release()
    out.release()

    if frame_times:
        avg_fps    = 1.0 / np.mean(frame_times)
        median_fps = 1.0 / np.median(frame_times)
        print("\n" + "="*50)
        print("  BASELINE PERFORMANCE SUMMARY")
        print("="*50)
        print(f"  Total frames : {frame_count}")
        print(f"  Average FPS  : {avg_fps:.2f}")
        print(f"  Median  FPS  : {median_fps:.2f}")
        print("="*50)


    print(f"Done! Baseline saved to {output_path}")

if __name__ == "__main__":
    # TODO: Replace with your video path or a sample video path
    video_source = "sample_drone.mp4" 
    run_baseline(video_source)