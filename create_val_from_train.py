# create_val_from_train.py
import os, random, shutil

# Source train that currently contains class folders
src_train = r"D:\PlantCareAi\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"

# Destination folders to create
out_train = r"D:\PlantCareAi\dataset\train"
out_val   = r"D:\PlantCareAi\dataset\val"

os.makedirs(out_train, exist_ok=True)
os.makedirs(out_val, exist_ok=True)

# Get class folders inside the source train
classes = [d for d in os.listdir(src_train) if os.path.isdir(os.path.join(src_train, d))]

for cls in classes:
    src_cls = os.path.join(src_train, cls)
    # create class folders in destination
    dst_train_cls = os.path.join(out_train, cls)
    dst_val_cls = os.path.join(out_val, cls)
    os.makedirs(dst_train_cls, exist_ok=True)
    os.makedirs(dst_val_cls, exist_ok=True)

    # list only files (skip nested folders)
    files = [f for f in os.listdir(src_cls) if os.path.isfile(os.path.join(src_cls, f))]
    if not files:
        continue

    random.shuffle(files)
    split_idx = max(1, int(0.8 * len(files)))  # ensure at least one file in train
    train_files = files[:split_idx]
    val_files = files[split_idx:]

    # copy files to new train and val folders
    for f in train_files:
        shutil.copy(os.path.join(src_cls, f), os.path.join(dst_train_cls, f))
    for f in val_files:
        shutil.copy(os.path.join(src_cls, f), os.path.join(dst_val_cls, f))

print("✅ Created dataset/train and dataset/val with 80/20 split.")
