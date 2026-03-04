from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

train_dir = r"D:\PlantCareAi\dataset\train"
val_dir   = r"D:\PlantCareAi\dataset\val"

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224,224), batch_size=32, class_mode='categorical')
val_gen   = val_datagen.flow_from_directory(val_dir,   target_size=(224,224), batch_size=32, class_mode='categorical')

print("Train samples:", train_gen.samples)
print("Val samples:", val_gen.samples)
print("Class mapping:", train_gen.class_indices)

# save mapping for inference
with open(r"D:\PlantCareAi\class_indices.json","w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class_indices.json")
