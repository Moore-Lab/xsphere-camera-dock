from pathlib import Path
import cv2

input_dir = Path(r"C:\Users\xbox\Desktop\xsphere-daq\xsphere-camera-dock\warmup_20260902")

# Find both .tif and .tiff files
images = sorted(
    list(input_dir.glob("*.tif")) +
    list(input_dir.glob("*.tiff"))
)

print("Directory:", input_dir)
print("Files found:", len(images))

if not images:
    print("\nNo TIFF files found!")
    print("Files actually in the directory:")
    for f in input_dir.iterdir():
        print("  ", f.name)
    raise SystemExit

# Read the first TIFF without converting its bit depth
image_path = images[0]
img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

if img is None:
    raise RuntimeError(f"Could not read: {image_path}")

print("\nFirst image:")
print("File:", image_path.name)
print("Data type:", img.dtype)
print("Minimum pixel value:", img.min())
print("Maximum pixel value:", img.max())
print("Image dimensions:", img.shape)
