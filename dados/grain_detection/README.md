---
license: cc-by-nc-4.0
pretty_name: Grain Detection Dataset (course release)
task_categories:
- object-detection
tags:
- agriculture
- grains
- seeds
- coco
- education
size_categories:
- 1K<n<10K
viewer: false
---

# Grain Detection Dataset (six classes)

An object-detection dataset of **small grains and seeds on a flat surface**: corn, peanuts, popcorn corn, pumpkin seeds, sunflower seeds and wheat.
Every image shows many objects of one type (from a few to several hundred per photo), and each object has a bounding box.
It is released for **education** (a course project on object detection); see License below.

- Task: object detection, 6 classes (category ids 0-5, fixed order below).
- Format: COCO (`bbox = [x, y, width, height]` in pixels, top-left origin).
- Size: 470 photos, 1,875 tiles, 27,611 tile-level boxes.

| id | name |
|---|---|
| 0 | corn |
| 1 | peanuts |
| 2 | popcorn_corn |
| 3 | pumpkin |
| 4 | sunflower |
| 5 | wheat |

## Folder structure

```
README.md  LICENSE.md  classes.json  splits.json  stats.json  checksums.sha256
coco/
  train/  images/*.jpg  labels.json     tile-level images + COCO labels (this is what a detector trains on)
  val/    images/*.jpg  labels.json
  test/   images/*.jpg  labels.json
photos/
  val/*.jpg   test/*.jpg                full photos of the val and test splits (not tiled)
annotations/
  val_photos.json  test_photos.json     COCO labels for those full photos
scripts/
  load_example.py   verify.py           tiny helpers (see below)
```

- **Tiles** (`coco/*/images`): photos were cut into tiles of at most 1500 x 1500 px with 20% overlap. Photos up to 1600 px on both sides are a single tile (the whole photo).
  Tile names are `<photo_name>_x<X>_y<Y>.jpg`, where `X, Y` is the tile's top-left corner in the photo. A box is kept in a tile only if at least 80% of its area is inside that tile and the clipped box is at least 8 px on each side.
  Some tiles have no boxes (kept as background examples, at most about 10% of a split).
- **Photos** (`photos/val`, `photos/test`): the full images, so you can evaluate at photo level (run your detector on tiles, merge the boxes back using the `x, y` in the tile name, compare with `annotations/*_photos.json`).
  Training photos are not provided as full images, only as tiles. Full-photo sizes: 76 images of about 1.5k x 1.5k px and 55 images of 3072 x 4080 or 4080 x 3072 px.
- Photo names are `<class>_<NNNN>.jpg`; the number is arbitrary and carries no meaning. `splits.json` maps each photo name to `train`, `val` or `test`.
- All images are JPEG without EXIF/colour-profile metadata. Pixel coordinates refer to the stored pixels (no rotation flags).

## Counts

Tile-level (what `coco/*/labels.json` contains). A grain lying in the overlap of two tiles, or cut by a tile border, can appear in more than one tile, so **box counts here are not counts of distinct grains**.

| class | split | photos | tiles | empty tiles | boxes | mean boxes / tile | median box w x h (px) | box side min / median / max (px) |
|---|---|---|---|---|---|---|---|---|
| corn | train | 58 | 102 | 2 | 2175 | 21.3 | 130 x 134 | 68 / 132 / 214 |
| corn | val | 8 | 96 | 4 | 903 | 9.4 | 139 x 129 | 75 / 134 / 195 |
| corn | test | 8 | 83 | 8 | 683 | 8.2 | 135 x 135 | 68 / 134 / 315 |
| peanuts | train | 55 | 99 | 3 | 1849 | 18.7 | 121 x 120 | 59 / 121 / 229 |
| peanuts | val | 11 | 33 | 3 | 236 | 7.2 | 124 x 130 | 80 / 128 / 234 |
| peanuts | test | 11 | 120 | 13 | 820 | 6.8 | 158 x 153 | 88 / 154 / 365 |
| popcorn_corn | train | 63 | 250 | 17 | 4288 | 17.2 | 89 x 89 | 40 / 89 / 162 |
| popcorn_corn | val | 11 | 55 | 2 | 983 | 17.9 | 84 x 86 | 46 / 87 / 102 |
| popcorn_corn | test | 10 | 21 | 2 | 605 | 28.8 | 84 x 80 | 39 / 84 / 116 |
| pumpkin | train | 81 | 411 | 11 | 4950 | 12.0 | 115 x 121 | 48 / 121 / 239 |
| pumpkin | val | 14 | 80 | 5 | 740 | 9.2 | 135 x 138 | 60 / 127 / 387 |
| pumpkin | test | 15 | 81 | 11 | 672 | 8.3 | 118 x 136 | 22 / 121 / 396 |
| sunflower | train | 33 | 165 | 11 | 2891 | 17.5 | 110 x 120 | 54 / 109 / 165 |
| sunflower | val | 14 | 25 | 2 | 559 | 22.4 | 112 x 119 | 54 / 107 / 138 |
| sunflower | test | 9 | 31 | 3 | 528 | 17.0 | 115 x 121 | 58 / 108 / 139 |
| wheat | train | 49 | 115 | 7 | 3276 | 28.5 | 72 x 74 | 37 / 69 / 167 |
| wheat | val | 7 | 73 | 2 | 579 | 7.9 | 86 x 81 | 60 / 82 / 150 |
| wheat | test | 13 | 35 | 0 | 874 | 25.0 | 77 x 74 | 44 / 71 / 123 |
| **all** | **train** | 339 | 1142 | | 19429 | 17.0 | | 37 / 105 / 239 |
| **all** | **val** | 65 | 362 | | 4000 | 11.0 | | 46 / 107 / 387 |
| **all** | **test** | 66 | 371 | | 4182 | 11.3 | | 22 / 113 / 396 |

Photo-level (`annotations/val_photos.json`, `annotations/test_photos.json`; one box per object, no duplicates):

| class | val photos | val boxes | test photos | test boxes | mean boxes / photo (val+test) |
|---|---|---|---|---|---|
| corn | 8 | 334 | 8 | 297 | 39.4 |
| peanuts | 11 | 191 | 11 | 344 | 24.3 |
| popcorn_corn | 11 | 506 | 10 | 480 | 47.0 |
| pumpkin | 14 | 395 | 15 | 432 | 28.5 |
| sunflower | 14 | 520 | 9 | 380 | 39.1 |
| wheat | 7 | 225 | 13 | 695 | 46.0 |

## How to load

With `pycocotools`:

```python
from pycocotools.coco import COCO
coco = COCO("coco/train/labels.json")
print(len(coco.getImgIds()), "tiles,", len(coco.getAnnIds()), "boxes")
img = coco.loadImgs(coco.getImgIds()[0])[0]            # img["file_name"] is relative to coco/train/images/
anns = coco.loadAnns(coco.getAnnIds(imgIds=img["id"]))  # each ann: bbox=[x,y,w,h], category_id in 0..5
```

With torchvision: `torchvision.datasets.CocoDetection("coco/train/images", "coco/train/labels.json")`.

Detector frameworks that expect a "COCO folder" (for example RF-DETR) need one folder per split with the images and one JSON file. Here the JSON is `labels.json` next to an `images/` folder;
if your framework wants the JSON inside the image folder (often named `_annotations.coco.json`), copy or symlink it there.

`scripts/load_example.py` prints the counts per class and saves an overlay with the boxes of one tile. `scripts/verify.py` checks the integrity of the release (ids, boxes inside images, no photo in two splits, counts, checksums).

## How the splits were made

Photos of the same sample group (consecutive shots of the same pile of grains, and photos with matching layouts) were kept together, and **whole groups** were assigned to train, val or test.
Every class has several groups in each split, and no photo or group appears in two splits.
The split is therefore by source photo and group, **never by tile**. Tiles of one photo overlap and look alike, and photos of one group look almost the same:
**do not re-split the tiles randomly** (or re-mix train/val/test), or your validation scores will be inflated by near-duplicates.
If you need a different split, group by photo name (and keep similar photos together) rather than by tile. `splits.json` gives the assignment per photo.

## Annotation notes (please read)

- The bounding boxes were **generated automatically with a vision-language model** (one box per object, on image tiles) and then **filtered with simple size rules**
  (boxes much smaller or much larger than the typical box of the same photo, or very elongated, were removed).
- **They were NOT manually verified or corrected.** Expect missed grains, doubled boxes, loose or tight boxes, and boxes merging two touching grains.
- Therefore any metric you compute on val/test measures **agreement with these automatic labels**, not accuracy against a human-verified ground truth. Report it that way.
- Tile-level boxes are clipped copies of the photo-level boxes (see the tile rule above); box widths and heights are in tile pixels.

## Known limitations

- One class per image: no photo mixes grain types, so a model can learn "background look -> class" shortcuts. Do not expect it to separate mixed piles.
- Unequal class sizes: pumpkin seeds and popcorn corn have the most photos, sunflower seeds the fewest (see the table).
- Small val/test: only about 65 photos each, from a handful of independent groups per class, so per-class scores are noisy.
- Limited variety: a few backgrounds, one lighting style and a limited range of viewpoints and distances. Models may not transfer to other backgrounds or lighting.
- Dense piles contain touching and overlapping grains, which are hard for both the labels and the models.

## Links
- Baseline model trained on this dataset: https://huggingface.co/niru-5/manama_course_rfdetr_grain_detection
- Course code (annotation app, training, weight/cost tasks): https://github.com/niru-5/manama_data_collection_course

## Authors
Niranjan Kukkala (niranjan.reddy.wo@gmail.com)

## Download
```bash
hf download niru-5/manama_course_dataset_grain_detection --repo-type dataset --local-dir grain_detection_public
python grain_detection_public/scripts/verify.py        # integrity check, prints "all checks passed"
```

## Intended use

Teaching and coursework in object detection (training, evaluating, error analysis, and discussing label quality). Not intended for any commercial use.

## License

[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/): you may share and adapt the data **for educational / non-commercial purposes**, with attribution.
See `LICENSE.md`.


## More information

More details about the dataset (collection protocol, annotation process, benchmarks) will be released after the course.
