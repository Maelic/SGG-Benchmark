# RelationDataset in COCO Format for Scene Graph Data

## Overview

The `RelationDataset` class is a new PyTorch dataset implementation that supports COCO format annotations with embedded relationship information. It preserves all functionality from the original `VGDataset` while working with a different data format. **Each JSON file represents one complete split (train, test, or val)**, eliminating the need for split logic within the dataset. It is somewhat similar to the format introduced in [OpenPSG](https://github.com/Jingkang50/OpenPSG).
It is intended to work with data collected through the [SGG-Annotate](https://github.com/Maelic/SGG-Annotate) codebase.

## COCO Format Structure

The expected JSON format includes:

### Categories
```json
"categories": [  // Original COCO format for objects
    {
        "id": 0,
        "name": "person",
        "supercategory": "none"
    },
    ...
],
"rel_categories": [ // New format for relations
    {
        "id": 0,
        "name": "above"
    },
    ...
]
```

### Annotations (Relations)
```json
"rel_annotations": [
    {
        "id": 0,
        "subject_id": 0,  // Correspond to the global id of the bounding box (see below)
        "predicate_id": 4, // rel_categories index (see above)
        "object_id": 1, // Correspond to the global id of the bounding box (see below)
        "image_id": 0
    },
    ...
]
```

### Annotations (Bounding Boxes)
```json
"annotations": [
    {
        "id": 0,
        "image_id": 0,
        "category_id": 5,
        "bbox": [214, 133, 124, 125],  // [x, y, width, height]
        "area": 15500,
        "iscrowd": 0
    },
    ...
]
```

## Usage

### Basic Usage
```python
from sgg_benchmark.data.datasets.dataloader import RelationDataset

# Create training dataset
train_dataset = RelationDataset(
    annotation_file='/path/to/train_relations.json',
    img_dir='/path/to/images'
)

# Create validation dataset  
val_dataset = RelationDataset(
    annotation_file='/path/to/val_relations.json',
    img_dir='/path/to/images'
)

# Create test dataset
test_dataset = RelationDataset(
    annotation_file='/path/to/test_relations.json',
    img_dir='/path/to/images'
)
```

### Parameters

- **annotation_file** (str): Path to COCO format JSON file with relationships (one file per split)
- **img_dir** (str): Directory containing images
- **transforms**: Image transformations (optional)
- **filter_empty_rels** (bool): Filter images without relationships
- **filter_duplicate_rels** (bool): Handle duplicate relationships
- **filter_non_overlap** (bool): Filter non-overlapping boxes
- **flip_aug** (bool): Enable horizontal flip augmentation
- **num_im** (int): Limit number of images (-1 for all)
- **custom_eval** (bool): Enable custom evaluation mode
- **custom_path** (str): Path for custom images

## Key Features Preserved

1. **Statistics Computation**: `get_statistics()` method computes foreground/background matrices, predicate frequencies, and triplet statistics
2. **Filtering Options**: Empty relationships, duplicates, non-overlapping boxes
3. **Data Augmentation**: Horizontal flipping support
4. **Custom Evaluation**: Support for custom image directories
5. **Native COCO Coordinates**: Uses original COCO bounding box coordinates without scaling
6. **Simplified Interface**: One JSON file per split eliminates split logic

## Differences from VGDataset

1. **Input Format**: Uses COCO JSON instead of HDF5 files
2. **Relationship Storage**: Relationships embedded in image data
3. **Box Format**: Handles COCO bbox format [x, y, w, h] → [x1, y1, x2, y2]
4. **Category Mapping**: Builds mappings from COCO categories
5. **Predicate Extraction**: Automatically extracts unique predicates from relationships
6. **No Box Scaling**: Uses original COCO coordinates directly (no BOX_SCALE normalization)
7. **No Split Parameter**: Each JSON file represents one complete split

## Compatibility

The RelationDataset is fully compatible with existing SGG-Benchmark training pipelines and produces the same output format as VGDataset, making it a drop-in replacement for COCO format data.
