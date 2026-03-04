# paste the training script here (exact content provided below)
# D:\PlantCareAi\train_model.py
import os, json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths and params
train_dir = r"D:\PlantCareAi\dataset\train"
val_dir   = r"D:\PlantCareAi\dataset\val"
model_dir = r"D:\PlantCareAi\models"
os.makedirs(model_dir, exist_ok=True)
checkpoint_path = os.path.join(model_dir, "best_model.h5")
final_path = os.path.join(model_dir, "plant_disease_model.h5")
log_csv = os.path.join(model_dir, "training_log.csv")

img_size = (224,224)
batch_size = 32   # reduce to 16 or 8 if OOM or very slow
initial_epochs = 20   # set low for smoke test; increase to 20+ for full training
fine_tune_epochs = 10  # set >0 after smoke test if you want fine-tuning
lr_head = 1e-4
lr_finetune = 1e-5

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=True)
val_gen   = val_datagen.flow_from_directory(val_dir,   target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

num_classes = len(train_gen.class_indices)
print("Num classes:", num_classes)

# Build model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(optimizer=Adam(learning_rate=lr_head), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks
callbacks = [
    ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    CSVLogger(log_csv)
]

# Load class weights if available
cw_path = r"D:\PlantCareAi\class_weights.json"
class_weight = None
if os.path.exists(cw_path):
    with open(cw_path,'r') as f:
        class_weight = json.load(f)
    print("Loaded class weights.")

# Stage 1: train head
history = model.fit(train_gen, validation_data=val_gen, epochs=initial_epochs, callbacks=callbacks, class_weight=class_weight)

# Optional Stage 2: fine-tune (only if fine_tune_epochs > 0)
if fine_tune_epochs > 0:
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model.compile(optimizer=Adam(learning_rate=lr_finetune), loss='categorical_crossentropy', metrics=['accuracy'])

    history_fine = model.fit(train_gen, validation_data=val_gen, epochs=initial_epochs + fine_tune_epochs,
                             initial_epoch=history.epoch[-1] if history.epoch else 0,
                             callbacks=callbacks, class_weight=class_weight)

# Save final model
model.save(final_path)
print("Training complete. Model saved to:", final_path)
with open(r"D:\PlantCareAi\class_indices.json","w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class_indices.json")
