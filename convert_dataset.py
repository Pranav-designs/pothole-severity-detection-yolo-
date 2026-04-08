import os
import xml.etree.ElementTree as ET
import shutil
import random

# Paths
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
ANNOTATIONS_DIR = os.path.join(DATASET_DIR, "annotations")

# Output directories
OUTPUT_DIR = "dataset_yolo"
TRAIN_IMAGES = os.path.join(OUTPUT_DIR, "train", "images")
TRAIN_LABELS = os.path.join(OUTPUT_DIR, "train", "labels")
VALID_IMAGES = os.path.join(OUTPUT_DIR, "valid", "images")
VALID_LABELS = os.path.join(OUTPUT_DIR, "valid", "labels")
TEST_IMAGES = os.path.join(OUTPUT_DIR, "test", "images")
TEST_LABELS = os.path.join(OUTPUT_DIR, "test", "labels")

# Class mapping
CLASS_MAPPING = {
    'minor_pothole': 0,   # LOW
    'medium_pothole': 1,  # MEDIUM
    'major_pothole': 2    # HIGH
}

# Create output directories
for dir_path in [TRAIN_IMAGES, TRAIN_LABELS, VALID_IMAGES, VALID_LABELS, TEST_IMAGES, TEST_LABELS]:
    os.makedirs(dir_path, exist_ok=True)

def convert_bbox(size, box):
    """Convert Pascal VOC bbox to YOLO format"""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

def convert_annotation(xml_file, output_label_file):
    """Convert single XML annotation to YOLO format"""
    tree = ET.parse(xml_file)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    with open(output_label_file, 'w') as f:
        for obj in root.findall('object'):
            class_name = obj.find('name').text.strip().lower()

            if class_name not in CLASS_MAPPING:
                print(f"⚠️  Unknown class: {class_name} in {xml_file}")
                continue

            class_id = CLASS_MAPPING[class_name]
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            xmax = float(bbox.find('xmax').text)
            ymin = float(bbox.find('ymin').text)
            ymax = float(bbox.find('ymax').text)

            bbox_converted = convert_bbox(
                (width, height),
                (xmin, xmax, ymin, ymax)
            )

            f.write(f"{class_id} {' '.join([str(round(x, 6)) for x in bbox_converted])}\n")

# Get all image files
all_images = [f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(all_images)

# Split dataset 80% train, 10% valid, 10% test
train_split = int(0.8 * len(all_images))
valid_split = int(0.9 * len(all_images))

train_images = all_images[:train_split]
valid_images = all_images[train_split:valid_split]
test_images = all_images[valid_split:]

print(f"📊 Dataset split:")
print(f"   Train: {len(train_images)} images")
print(f"   Valid: {len(valid_images)} images")
print(f"   Test:  {len(test_images)} images")

def process_split(image_list, images_output, labels_output, split_name):
    success = 0
    skipped = 0
    for img_file in image_list:
        # Get corresponding annotation
        base_name = os.path.splitext(img_file)[0]
        xml_file = os.path.join(ANNOTATIONS_DIR, base_name + '.xml')

        if not os.path.exists(xml_file):
            print(f"⚠️  No annotation for {img_file}")
            skipped += 1
            continue

        # Copy image
        src_img = os.path.join(IMAGES_DIR, img_file)
        dst_img = os.path.join(images_output, img_file)
        shutil.copy2(src_img, dst_img)

        # Convert annotation
        dst_label = os.path.join(labels_output, base_name + '.txt')
        convert_annotation(xml_file, dst_label)
        success += 1

    print(f"✅ {split_name}: {success} processed, {skipped} skipped")

# Process each split
process_split(train_images, TRAIN_IMAGES, TRAIN_LABELS, "Train")
process_split(valid_images, VALID_IMAGES, VALID_LABELS, "Valid")
process_split(test_images, TEST_IMAGES, TEST_LABELS, "Test")

# Create data.yaml
yaml_content = f"""train: ../dataset_yolo/train/images
val: ../dataset_yolo/valid/images
test: ../dataset_yolo/test/images

nc: 3
names: ['minor_pothole', 'medium_pothole', 'major_pothole']

# Severity mapping:
# 0 = minor_pothole  (LOW severity)
# 1 = medium_pothole (MEDIUM severity)
# 2 = major_pothole  (HIGH severity)
"""

with open(os.path.join(OUTPUT_DIR, "data.yaml"), 'w') as f:
    f.write(yaml_content)

print(f"\n✅ data.yaml created!")
print(f"✅ Dataset conversion complete!")
print(f"📁 Output: {OUTPUT_DIR}/")