# nnInteractive Lesion Segmentation Validation

This repository provides tools to generate simulated interactive prompts (points, bounding boxes, diameter lines, spline scribbles) and reference inference sessions for two nnInteractive backends (v1 and v2). It is intended for research, validation, and reproducible experiments with interactive lesion segmentation.

## Repository Layout

Top-level structure:

- `interactive_seg/` — core Python package
  - `config/` — pydantic config classes for prompt generation and ROI cropping
  - `prompt_generation/` — functions that build prompts from label masks (`generate_nninteractive_prompts`)
  - `utils/` — geometry transforms, helpers, and prompt primitives (points, boxes, diameter, spline)
  - `io/` — NIfTI load/save and prompt serialization helpers
  - `device/` — `get_device()` helper
  - `seg_model/` — base `InteractiveSegmentationModel` class
- `interactive_seg_nninteractive_v1/` — v1 compatibility wrapper (config, model, session)
- `interactive_seg_nninteractive_v2/` — v2 compatibility wrapper (config, model, session)
- `model_weights/` — expected location for downloaded or manually placed model checkpoints

## Design Notes

- **Config-driven**: prompt generation uses `pydantic` models to validate and document parameters.
- **Deterministic prompts**: set `seed` in `NNInteractivePromptGenerationConfigBase` for reproducibility.
- **Minimal top-level exports**: only the most commonly used classes/functions are exported at the package root; use subpackage imports (e.g. `interactive_seg_nninteractive_v1.config` or `interactive_seg_nninteractive_v2.config`) for more granular access.

## Getting Started

### 1. Create and activate a `uv` virtual environment

```bash
uv venv venv_val_nninteractive --python 3.14
source venv_val_nninteractive/bin/activate  # on Windows: venv_val_nninteractive\Scripts\activate
```

### 2. Install Python requirements

```bash
# install this package in editable mode
uv pip install -e .

# install huggingface_hub if you plan to use automatic v2 model download
uv pip install huggingface_hub
```

> **Note:** `torch` is a required dependency for running inference (not needed for prompt generation alone). Install the build appropriate for your platform (CPU, CUDA, or MPS) by following the [official PyTorch install guide](https://pytorch.org/get-started/locally/).

## Examples

### Example 1 — Generate prompts only (fast, no heavy models required)

This example only exercises the prompt-generation utilities, so it runs quickly on CPU without downloading any model weights.

```python
import numpy as np
from interactive_seg import (
    NNInteractivePromptGenerationConfig,
    NNInteractivePromptGenerationConfigBase,
    generate_nninteractive_prompts,
)
from interactive_seg_nninteractive_v2.config import NNInteractiveV2ModelConfig

# Replace with real image/mask loading in real use
mask = np.zeros((64, 64, 32), dtype=np.uint8)
mask[20:40, 20:40, 10:20] = 1

model_cfg = NNInteractiveV2ModelConfig()  # only affects device/download behavior for generators

prompt_cfg = NNInteractivePromptGenerationConfigBase(
    seed=42,
    prompt_generation_config=NNInteractivePromptGenerationConfig(
        num_pts_pos=1,
        num_bbox_pos=1,
    ),
)

prompts = generate_nninteractive_prompts(config=prompt_cfg, model_config=model_cfg, mask=mask)
print("Generated prompt keys:", list(prompts.keys()))
```

### Example 2 — Full inference (v2 wrapper)

This example requires model weights and `torch`. Replace the synthetic image/mask with real data loading in practice.

```python
import numpy as np
from interactive_seg import (
    NNInteractivePromptGenerationConfig,
    NNInteractivePromptGenerationConfigBase,
    generate_nninteractive_prompts,
)
from interactive_seg_nninteractive_v2 import NNInteractiveV2ModelConfig, NNInteractiveV2InferenceSession

# Replace with real image/mask loading in real use
img = np.zeros((64, 64, 32), dtype=np.float32)
mask = np.zeros((64, 64, 32), dtype=np.uint8)
mask[20:40, 20:40, 10:20] = 1

# Point to local model_weights or allow automatic download
model_cfg = NNInteractiveV2ModelConfig(download_dir="./model_weights/")

prompt_cfg = NNInteractivePromptGenerationConfigBase(
    seed=0,
    prompt_generation_config=NNInteractivePromptGenerationConfig(num_pts_pos=1, num_bbox_pos=1),
)

prompts = generate_nninteractive_prompts(prompt_cfg, model_cfg, mask)

session = NNInteractiveV2InferenceSession(config=model_cfg)  # may download/init model
session.set_image(img)
session.run(prompts_dict=prompts, labels=None)
output = session.output
print("Output unique labels:", np.unique(output))
```

## Model Weights and Downloads

- `interactive_seg_nninteractive_v1.model` and `interactive_seg_nninteractive_v2.model` use `huggingface_hub.snapshot_download` to fetch the `nnInteractive_v1.0` snapshot by default. To avoid downloads at runtime, pre-download the model into `model_weights/` and set `NNInteractiveV1ModelConfig(download_dir="./model_weights/")` or `NNInteractiveV2ModelConfig(download_dir="./model_weights/")`

## Troubleshooting & Notes

- **CUDA / MPS import or build errors**: ensure `torch` is installed with the correct backend for your system (see the PyTorch install guide linked above).
- **Hugging Face download failures**: check network access and, if the target repo is private, that you've supplied a valid auth token (e.g. via `huggingface-cli login` or the `HF_TOKEN` environment variable).
