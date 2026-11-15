# AI-generated image detection

CSCI 594: AI-generated image detection on CIFAKE, comparing a custom CNN
trained from scratch with pretrained ConvNeXt V2-Tiny. The scratch CNN has
separate experiments with and without data augmentation.

## Setup

Use Python 3.11 or 3.12 with a compatible PyTorch installation. From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

This command installs CPU packages. For CUDA, use the matching command from the
[official PyTorch installer](https://pytorch.org/get-started/locally/).

## Data

Download [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
as `archive.zip` in this folder, then run:

```bash
python prepare_data.py
```

Preparation checks every image and extracts the archive into `data/cifake/`.
It keeps the supplied 20,000-image test set intact, removes training images
identical to test images, and keeps one copy of each remaining training image.
Duplicates are identified by SHA256 of decoded RGB pixels, not JPEG file bytes.
The remaining training pool is split 80/20 by class using Python's random seed 42.

`splits/cifake.csv` records every selected image's path, label, and split.
`splits/data_audit.json` records counts, excluded duplicates, and the archive hash.
Preparation refuses to overwrite existing prepared data or splits. Duplicate
images within the original test set are retained and counted in the audit.
Near-duplicates are not detected by this check.

Labels are explicitly **REAL=0, FAKE=1**. Training and evaluation read the saved
CSV; loading the extracted folders directly would bypass the exclusions.

Dataset usage conditions and required citations are documented in the
[authors' repository](https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images).
Do not commit the archive, extracted images, environment, or checkpoints.
Small experiment records (configs, training histories, and test metrics) are
tracked under `runs/`; checkpoint files remain local.

## Model and training

The model has three 3x3 convolutional layers with 32, 64, and 128 channels.
Each is followed by ReLU and 2x2 max pooling. Global average pooling and a
128-to-2 linear layer produce raw logits. There are 93,506 trainable parameters.
Images remain 32x32 RGB and are scaled to [-1, 1].

Train a one-epoch timing pilot first:

```bash
python train.py --epochs 1 --output runs/cnn-pilot
```

Then start a fresh final run:

```bash
python train.py --epochs 20 --output runs/cnn-baseline
```

For the augmentation comparison, start another fresh run:

```bash
python train.py --epochs 20 --augment --output runs/cnn-augmented
```

Augmentation randomly crops a 32x32 region after four pixels of reflection
padding on each side, then flips horizontally with probability 0.5. Transforms
are sampled each time an image is loaded; no extra image files are created.
Only training images are augmented. Validation and test preprocessing remain
deterministic. The architecture, split, optimizer, and default hyperparameters
are the same as the baseline. The config records whether augmentation was
enabled and the exact training transform.

Defaults: batch size 64, Adam, learning rate 0.001, weight decay 0.0001,
training seed 42, and six CPU threads. CUDA is used if available. Each output
directory must be new, to prevent overwriting an experiment. The final run starts
from random weights; it does not resume the pilot. Training does not load test data.

Each run saves `config.json`, per-epoch loss/accuracy/F1 and timing in `history.csv`,
and `best.pt`, selected by validation F1 with FAKE as the positive class.
The config includes library versions and the data audit. Deterministic operations
are requested; identical results across hardware or library versions are not guaranteed.

After model selection, evaluate the selected checkpoint:

```bash
python evaluate.py runs/cnn-baseline/best.pt
```

For the augmented model, use `runs/cnn-augmented/best.pt` instead. Select the
checkpoint using validation F1 before evaluating it on the test set.

Evaluation runs on CPU and saves `test_metrics.json` with loss, accuracy,
precision, recall, F1, and a confusion matrix. Rows are true labels and columns
are predictions, both ordered REAL, FAKE. Do not tune the model using test results.

## ConvNeXt V2-Tiny on an NVIDIA GPU

Keep the CPU environment and create a separate GPU environment with Python 3.12:

```bash
.venv/bin/python -m venv .venv-gpu
.venv-gpu/bin/python -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu132
.venv-gpu/bin/python -m pip install -r requirements-convnext.txt
.venv-gpu/bin/python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

These CUDA 13.2 packages target the project's RTX 5050 laptop GPU and its existing
NVIDIA 595.99.02 driver. Other systems should choose compatible packages from the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
No system CUDA toolkit is required for these prebuilt packages. GPU commands
must run in a context that can access the GPU; a restricted agent sandbox may
hide GPU devices even when training from the user's terminal works.

The model is `convnextv2_tiny.fcmae_ft_in1k`, pretrained using FCMAE and fine-tuned
on ImageNet-1K. Its classification head is replaced with two outputs. See the
[model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in1k) and
[paper](https://arxiv.org/abs/2301.00808).
Weights download into the ignored `data/.cache/` directory.

First run a one-epoch pilot:

```bash
.venv-gpu/bin/python -u train.py --model convnextv2-tiny --device cuda --amp --augment --batch-size 16 --workers 2 --epochs 1 --output runs/convnextv2-pilot
```

The recorded RTX 5050 pilot completed one training/validation epoch in 611.6
seconds (10.2 minutes), with 96.849% validation accuracy and validation F1 0.9687.
At that rate, 20 epochs would take about 3 hours 25 minutes, excluding setup;
actual runtime can vary. This is a preliminary validation result, not a final
test result. The earlier custom-CNN timings were measured on CPU.

Then fine-tune for 20 epochs in a new directory:

```bash
.venv-gpu/bin/python -u train.py --model convnextv2-tiny --device cuda --amp --augment --batch-size 16 --workers 2 --epochs 20 --output runs/convnextv2
```

All model parameters are fine-tuned with Adam, learning rate 0.0001 and weight
decay 0.0001. The run starts from pretrained weights; it does not resume the
pilot. CUDA mixed precision reduces memory usage. `--device cuda` fails clearly
if CUDA is unavailable rather than silently running this experiment on CPU.

The saved split and label mapping remain unchanged. Native 32x32 images receive
the same training-only crops/flips used by the augmented CNN, then are resized
to 224x224 using bicubic interpolation and normalized with the pretrained
model's mean/std. Validation and test use deterministic resizing and normalization.
Upsampling adds no image detail. This compares full modeling approaches, with
differences in architecture, pretraining, preprocessing, and training settings.

After selecting the final checkpoint using validation F1:

```bash
.venv-gpu/bin/python evaluate.py runs/convnextv2/best.pt --device cuda --batch-size 16
```

Evaluation restores the architecture and preprocessing from the checkpoint.
Existing custom-CNN checkpoints remain supported. It defaults to CPU; CUDA
evaluation uses mixed precision when the checkpoint was trained with `--amp`.

## Recorded CIFAKE results

Both experiments use seed 42, the same saved splits and hyperparameters, and
the 20,000-image official test set. Checkpoints were selected by validation F1;
epoch 20 was selected in both runs. FAKE is the positive class.

| Model | Test accuracy | Test precision | Test recall | Test F1 | Training + validation time |
|---|---:|---:|---:|---:|---:|
| CNN without augmentation | 95.745% | 95.924% | 95.550% | 95.737% | 16.0 min |
| CNN with augmentation | 95.645% | 95.622% | 95.670% | 95.646% | 18.2 min |

Augmentation reduced test accuracy by 0.10 percentage points and F1 by about
0.09 percentage points in this run. The small difference from one seed does
not establish a consistent effect. The original test set retains 38 duplicate
images, as documented in the data audit; identical-image overlap between splits
was removed. External-dataset generalization has not been evaluated.

Full results: [baseline](runs/cnn-baseline/test_metrics.json) and
[augmented CNN](runs/cnn-augmented/test_metrics.json). Training histories and
configs are stored alongside these files. Keep the test set for reporting;
select any further hyperparameters using validation data.

## Checks

```bash
python -m unittest discover -s tests -v
```

Checks cover split reproducibility and leakage, learning a tiny synthetic sample,
checkpoint round trips, positive-class metrics, and augmentation restricted to
training images with labels preserved. Synthetic learning checks
implementation behavior; real CIFAKE performance must come from an actual run.
For ConvNeXt and GPU checks, run the same test command using
`.venv-gpu/bin/python`. Optional ConvNeXt tests are skipped if `timm` is absent;
the CUDA training-step test is skipped when a GPU is inaccessible.
