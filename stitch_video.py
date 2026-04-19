import cv2
import os

sequence_dir = '/Users/KAVITA/projects/AERIAL GUARDIAN/drone_tracker/VisDrone2019-MOT-val/sequences/uav0000086_00000_v'

# output_path = '/Users/KAVITA/projects/AERIAL GUARDIAN/drone_tracker/drone_test.mp4'
output_path = '/Users/KAVITA/projects/AERIAL GUARDIAN/drone_tracker/drone_test.avi'

images = [img for img in os.listdir(sequence_dir) if img.endswith('.jpg')]
images.sort()

if not images:
    print("No images found.")
    exit()

first_image = cv2.imread(os.path.join(sequence_dir, images[0]))
h, w, layers = first_image.shape

"""
# OLD
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
"""

# NEW — use DIVX on Windows, save as .avi (most reliable on Windows)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(output_path.replace('.mp4', '.avi'), fourcc, 30.0, (w, h))

for img_name in images:
    img_path = os.path.join(sequence_dir, img_name)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(f"Saved video to {output_path}")