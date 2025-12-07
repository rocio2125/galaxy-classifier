# %%
import os
import h5py
import numpy as np
import cv2

# %%
#   CARGA DE HDF5
# ==============================
input_h5 = os.path.join("data", "Galaxy10_DECals_NoDuplicated.h5")
output_dir = "reduced_h5"
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = 128

h5 = h5py.File(input_h5, "r")
images = h5["images"]
labels = h5["ans"]
N = images.shape[0]
print("Total imágenes:", N)

# %%
#   SPLIT train/val/test
# ==============================
indices = np.arange(N)
np.random.seed(42)
np.random.shuffle(indices)

train_end = int(0.7 * N)
val_end   = int(0.85 * N)

splits = {
    "train": indices[:train_end],
    "val":   indices[train_end:val_end],
    "test":  indices[val_end:]
}

print(f"splits[train]: {len(splits['train'])}, splits['Val']: {len(splits['val'])}, splits['Test']: {len(splits['test'])}")

# %%
# FUNCIÓN DE REDIMENSIONADO
# =============================
def resize_img(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

# %%
# FUNCIÓN DE REDIMENSIONADO
# =============================
def resize_img(img):
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

# =============================
# CREAR LOS NUEVOS ARCHIVOS H5
# =============================
for split_name, split_idx in splits.items():
    out_path = os.path.join(output_dir, f"{split_name}_{IMG_SIZE}.h5")

    file_out = h5py.File(out_path, "w")
    d_imgs = file_out.create_dataset(
        "images", 
        (len(split_idx), IMG_SIZE, IMG_SIZE, 3),
        dtype=np.uint8
    )
    d_lbls = file_out.create_dataset(
        "labels", 
        (len(split_idx),),
        dtype=np.int32
    )

    print(f"Generando {split_name}: {len(split_idx)} elementos...")

    for i, orig_i in enumerate(split_idx):
        img = images[orig_i]              # sin cargar a RAM
        img_resized = resize_img(img)     # reduce tamaño
        d_imgs[i] = img_resized
        d_lbls[i] = labels[orig_i]

        if i % 2000 == 0:
            print(f"{i}/{len(split_idx)} procesadas...")

    file_out.close()
    print(f"{split_name} guardado en: {out_path}")

h5.close()
print("\nProceso finalizado.")


