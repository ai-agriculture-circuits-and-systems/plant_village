# PlantVillage Dataset

This project utilizes the PlantVillage dataset, which contains images of healthy and unhealthy plant leaves.

## Description

The PlantVillage dataset consists of 54,303 healthy and unhealthy leaf images divided into 38 categories based on species and disease. The images are sourced from the [PlantVillage project](https://plantvillage.psu.edu/).

**Note:** The dataset used here is obtained via TensorFlow Datasets (TFDS), sourced from a republished version available on Mendeley Data, as the original dataset link from plantvillage.org might not be directly accessible for download in this format. Images labeled as `Background_without_leaves` were excluded as they were not part of the original dataset described in the paper.

Original Paper: [An open access repository of images on plant health...](https://arxiv.org/abs/1511.08060)
Dataset Repository (used by TFDS): [Mendeley Data](https://data.mendeley.com/datasets/tywbtsjrjv/1)

## Dataset Details (via TFDS)

*   **Version**: `1.0.2` (default in TFDS)
*   **Download Size**: ~827.82 MiB
*   **Dataset Size**: ~815.37 MiB
*   **Splits**:
    *   `train`: 54,303 examples
*   **Features**:
    *   `image`: `tf.Tensor` of shape `(None, None, 3)` and dtype `uint8`.
    *   `label`: `tf.ClassLabel` (int64) representing the 38 classes.
    *   `image/filename`: `tf.Tensor` (string) with the original filename.
*   **Supervised Keys**: `('image', 'label')`

## How to Load

You can load this dataset using the `tensorflow_datasets` library:

```python
import tensorflow_datasets as tfds
import tensorflow as tf

# Configure TFDS data directory if needed (optional)
# tfds.core.utils.gcs_utils._is_gcs_disabled = True # Uncomment if GCS access is blocked
# data_dir = "/path/to/your/data" # Specify a local directory
# ds, info = tfds.load('plant_village', split='train', shuffle_files=True, data_dir=data_dir, with_info=True, as_supervised=True)

# Load the PlantVillage dataset (downloads automatically if not cached)
ds, info = tfds.load('plant_village', split='train', shuffle_files=True, with_info=True, as_supervised=True)

# Print dataset info
print(info)

# Basic dataset preparation example
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224)) # Resize images
    image = tf.cast(image, tf.float32) / 255.0  # Normalize images
    return image, label

ds = ds.map(preprocess)
ds = ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

# Iterate over the first batch
for images, labels in ds.take(1):
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

```

## Citation

If you use this dataset, please cite the original paper:

```bibtex
@article{DBLP:journals/corr/HughesS15,
  author    = {David P. Hughes and
               Marcel Salath{\'{e}} },
  title     = {An open access repository of images on plant health to enable the
               development of mobile disease diagnostics through machine
               learning and crowdsourcing},
  journal   = {CoRR},
  volume    = {abs/1511.08060},
  year      = {2015},
  url       = {http://arxiv.org/abs/1511.08060},
  archivePrefix = {arXiv},
  eprint    = {1511.08060},
  timestamp = {Mon, 13 Aug 2018 16:48:21 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/HughesS15},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
``` 

## Licensing

The dataset images are generally associated with Creative Commons licenses. This project assumes usage under **CC BY 3.0** as specified. Please refer to the original sources for specific licensing details. The TensorFlow Datasets library code itself is under Apache 2.0.