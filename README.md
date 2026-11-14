# DeOR — training-set manifest and reproducibility materials (anonymous)

This repository accompanies the author response for ICLR 2027 Submission 43280.
It provides the exact training-image manifest, the filtering script that produces it from the
public Pix2Cap-COCO annotations, and the target-feature normalization statistics.
Training/evaluation code and checkpoints will be released with the camera-ready version.

## Contents

| Path | Description |
|---|---|
| `manifests/train_image_ids.txt` | The 16,477 COCO train2017 image IDs used for training (one per line, manifest order). |
| `manifests/train_manifest.jsonl.gz` | Per-image training targets: object sequence (`caption`), per-object Pix2Cap segment IDs, categories, descriptions, and boxes; image/panoptic paths are relative to the COCO root. |
| `manifests/filter_stats.json` | Exclusion counts produced by the filter (see below). |
| `preprocess/prepare_pix2cap_thing.py` | Filter that builds the manifest from `pix2cap_coco_train.json`. |
| `preprocess/pix2cap_filters.py` | Caption-validity patterns and crop-visibility test used by the filter. |
| `stats/siglip2_target_feature_norm_stats.pt` | Channel-wise `running_mean` / `running_var` (1,152-d) for the frozen SigLIP2 target features (Appendix A.7). |

## Reproducing the 16,477-image training set

```bash
pip install transformers
python preprocess/prepare_pix2cap_thing.py \
    --coco_root <COCO_ROOT> --split train \
    --output train_manifest.jsonl
# expects <COCO_ROOT>/pix2cap_coco_train.json, <COCO_ROOT>/train2017/, <COCO_ROOT>/annotations/panoptic_train2017/
```

`pix2cap_coco_train.json` used in our experiments: SHA-256 `113ed196b9e6ee483aea65e325f57810f2849e330a207fb481ec48a9a0b35912`.
The tokenizer (`nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B`, with the four special tokens added) is used only to
check the 1,024-token sequence limit, which no retained image exceeds.

### Filtering rule

Each image is resized so that its shorter side is 512 px and center-cropped to 512×512, matching training.
A COCO *thing* instance is **crop-visible** if its box intersects the crop. An image is kept only if
**every** crop-visible thing has a valid Pix2Cap description; stuff regions are excluded from the sequence
and become the register (background) target.

| Stage | Images |
|---|---|
| Pix2Cap-COCO training images | 18,212 |
| Excluded: a crop-visible thing has an invalid description | −1,509 |
| &nbsp;&nbsp;– description states the object is not visible | 1,457 |
| &nbsp;&nbsp;– description says the object cannot be described/located/identified | 44 |
| &nbsp;&nbsp;– apology text | 8 |
| Excluded: no thing instance remains inside the crop | −226 |
| **Training set** | **16,477** |

Each excluded image is counted under its first invalid crop-visible instance. Because visibility and validity
are evaluated inside the 512×512 crop, a filter applied to the full image yields a different subset.

Checksums: `train_image_ids.txt` SHA-256 `a63cc71c756b801b06efa3c277191c35d76728acfa9af9e696094dfed0883fae`.

## Other settings (see Appendix A.7 and B.2–B.3 of the revised paper)

- Training seed 42 (no separate data-sampling seed); 5,000 optimizer steps for DeOR-B and DeOR-L.
- Pretrained initializations: `nyu-visionx/Scale-RAE-Qwen1.5B_DiT2.4B` (DeOR-B), `nyu-visionx/Scale-RAE-Qwen7B_DiT9.8B` (DeOR-L);
  encoders `google/siglip2-so400m-patch16-512` (source) and `google/siglip2-so400m-patch14-224` (target).
- Decomposition inference: greedy decoding, 512-token budget, at most 50 objects, 512×512 center crop.
