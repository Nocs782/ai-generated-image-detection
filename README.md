# AI-generated image detection

CSCI 594: AI-generated image detection on CIFAKE, comparing a custom CNN
trained from scratch with pretrained ConvNeXt V2-Tiny and MambaVision-T.
The scratch CNN has separate experiments with and without data augmentation.

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

## MambaVision-T on an NVIDIA GPU

MambaVision-T is a hybrid Mamba/Transformer vision model. Use a separate Python
3.12 environment so its dependency versions do not affect the working ConvNeXt
environment:

```bash
.venv/bin/python -m venv .venv-mamba
.venv-mamba/bin/python -m pip install --timeout 120 torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
.venv-mamba/bin/python -m pip install --timeout 120 -r requirements-mamba.txt \
  "https://github.com/state-spaces/mamba/releases/download/v2.3.2.post1/mamba_ssm-2.3.2.post1%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
.venv-mamba/bin/python -m pip check
```

The install command supplies the official `mamba-ssm` CUDA 13 wheel for
PyTorch 2.10, Python 3.12, Linux x86-64, and the C++11 ABI. It includes the
selective-scan CUDA extension, so no local CUDA compilation is needed.
Other platforms require a matching wheel rather than this environment's file.

We use [NVIDIA's official Hugging Face model](https://huggingface.co/nvidia/MambaVision-T-1K),
pinned to revision `b1de77e17599566d98efb701c0231b1095dc3a67`. Transformers loads
the reviewed model code from that immutable revision and safetensors weights.
The underlying PyTorch model is retained and its ImageNet head is replaced
with a two-class linear layer. This avoids the pip `mambavision` package's old
dependency pins without modifying NVIDIA's model code. Weights and downloaded
model code stay under the ignored `data/.cache/` directory.

Start with a one-epoch pilot:

```bash
.venv-mamba/bin/python -u train.py --model mambavision-t --device cuda --amp --augment --batch-size 16 --workers 2 --epochs 1 --output runs/mambavision-pilot
```

The recorded RTX 5050 pilot completed one training/validation epoch in 324.4
seconds (5.4 minutes), with 96.677% validation accuracy and validation F1 0.9669.
At that pace, 20 epochs would take about 1 hour 48 minutes, excluding setup;
actual runtime can vary. These are preliminary validation results. The pilot
does not evaluate the official test set.

Then start a fresh 20-epoch run:

```bash
.venv-mamba/bin/python -u train.py --model mambavision-t --device cuda --amp --augment --batch-size 16 --workers 2 --epochs 20 --output runs/mambavision
```

This fine-tunes all parameters with Adam, learning rate 0.0001, weight decay
0.0001, and CUDA mixed precision. It uses the same saved splits, training-only
augmentation, 224x224 resizing, and ImageNet normalization as ConvNeXt. The
config records the model revision, Mamba/Transformers versions, and the model's
default stochastic-depth rate. Seeds and PyTorch deterministic settings are
recorded/requested; custom CUDA kernels do not guarantee bitwise reproducibility.

After model selection by validation F1, evaluate with CUDA:

```bash
.venv-mamba/bin/python evaluate.py runs/mambavision/best.pt --device cuda --batch-size 16
```

The CUDA selective-scan kernel is required for both training and evaluation.
The test set is not used in the pilot or for checkpoint selection.

## External testing on GenImage

Download `genimage_test.zip` from the
[GenImage community mirror](https://huggingface.co/datasets/jzousz/GenImage/tree/main)
and place it in the project root alongside the original CIFAKE `archive.zip`.
The downloaded archive contains 100,000 images across eight generators.
Use the original [GenImage project](https://github.com/GenImage-Dataset/GenImage)
for citation and dataset terms. Raw images and archives stay out of Git.

Prepare the external sample once:

```bash
.venv/bin/python -u prepare_genimage.py
```

Preparation selects 625 real and 625 generated images per generator, using
seed 42 and a sorted/shuffled candidate list: 10,000 images total. Only the
selected images are extracted into `data/genimage-test/`. The script checks
selected members' ZIP CRCs, decodes them, excludes identical RGB images across
the sample, and rejects matches against all original CIFAKE images at native
resolution or after bicubic resizing to 32x32. Rejected candidates are replaced
until each generator/class reaches its fixed count. It refuses to overwrite
existing extracted data. On a fresh clone with tracked manifests but no raw
images, it regenerates the sample and requires an exact match with the saved
manifest/audit before keeping the extracted images.

The tracked `splits/genimage.csv` lists every selected path, label, generator,
and decoded-pixel digest. `splits/genimage_audit.json` records the archive and
manifest SHA256, selection counts, exclusions, native sizes, and file formats.
Image and manifest integrity are checked again during evaluation.

Evaluate the four validation-selected checkpoints without further training:

```bash
.venv/bin/python -u evaluate.py runs/cnn-baseline/best.pt --dataset genimage --batch-size 64
.venv/bin/python -u evaluate.py runs/cnn-augmented/best.pt --dataset genimage --batch-size 64
.venv-gpu/bin/python -u evaluate.py runs/convnextv2/best.pt --dataset genimage --device cuda --batch-size 16
.venv-mamba/bin/python -u evaluate.py runs/mambavision/best.pt --dataset genimage --device cuda --batch-size 16
```

Each run writes `genimage_metrics.json` next to its checkpoint, preserving
`test_metrics.json` for CIFAKE. Results include overall and per-generator
accuracy, precision, recall, F1, loss, and confusion matrices, plus the checkpoint
SHA256 and sample audit. Overall metrics are computed from summed confusion
matrices; overall loss is weighted by image count.

External images are converted to RGB and resized directly to the checkpoint's
input size (32x32 for the custom CNNs, 224x224 for the pretrained models), with
the checkpoint's normalization and no augmentation. Already-32x32 CIFAKE images
retain exactly the same preprocessing values. All models use the same external
manifest; do not choose checkpoints, thresholds, or hyperparameters using these
external results.

This tests combined shifts in content, generator, native resolution, and
compression. Real images in this archive are JPEG and generated images are PNG;
decoding and resizing do not erase compression traces. GenImage's ImageNet
origins may overlap with pretrained-model exposure. Exact-pixel checks do not
rule out near duplicates or overlap in pretraining data.

## Recorded CIFAKE results

All four experiments use seed 42, the same saved splits, and the 20,000-image
official test set. Checkpoints were selected by validation F1: epoch 20 for
both custom-CNN runs and MambaVision-T, and epoch 11 for ConvNeXt V2. FAKE is
the positive class.
Model-specific training settings are recorded in each run's config.

| Model | Device | Test accuracy | Test precision | Test recall | Test F1 | Training + validation time |
|---|---|---:|---:|---:|---:|---:|
| CNN without augmentation | CPU | 95.745% | 95.924% | 95.550% | 95.737% | 16.0 min |
| CNN with augmentation | CPU | 95.645% | 95.622% | 95.670% | 95.646% | 18.2 min |
| Pretrained ConvNeXt V2-Tiny with augmentation | RTX 5050 | 98.195% | 98.046% | 98.350% | 98.198% | 197.8 min |
| Pretrained MambaVision-T with augmentation | RTX 5050 | 97.430% | 96.998% | 97.890% | 97.442% | 111.8 min |

Augmentation reduced test accuracy by 0.10 percentage points and F1 by about
0.09 percentage points in this run. The small difference from one seed does
not establish a consistent effect.

ConvNeXt V2 improved test accuracy by 2.45 percentage points over the scratch
baseline, with 361 errors versus 851. Its selected epoch-11 checkpoint achieved
98.242% validation accuracy. Validation F1 was lower at epoch 20, so the saved
best checkpoint was used for test evaluation. These runs compare complete
modeling approaches; architecture, pretraining, input resolution, and training
settings differ. The timings also use different hardware.

MambaVision-T completed all 20 epochs in 111.8 minutes. Its selected epoch-20
checkpoint achieved 97.695% validation accuracy and 97.430% test accuracy,
with 514 test errors. Test accuracy was 1.685 percentage points above the
scratch CNN and 0.765 percentage points below ConvNeXt V2. MambaVision took
less training/validation time on the same GPU in these runs; software versions
and model defaults differ. Single-seed results do not establish a consistent
advantage across runs or datasets.

The original CIFAKE test set retains 38 duplicate images, as documented in the
data audit; identical-image overlap between splits was removed. External
GenImage results are reported separately below.

Full results: [baseline](runs/cnn-baseline/test_metrics.json),
[augmented CNN](runs/cnn-augmented/test_metrics.json),
[ConvNeXt V2](runs/convnextv2/test_metrics.json), and
[MambaVision-T](runs/mambavision/test_metrics.json). Training histories and
configs are stored alongside these files. Keep the test set for reporting;
select any further hyperparameters using validation data.

## Recorded GenImage external results

All four frozen checkpoints were evaluated on the same 10,000-image external
sample: 5,000 REAL and 5,000 FAKE, with 625 per class for each of eight
generators. Checkpoints were selected using CIFAKE validation F1 before this
evaluation. No external images were used for training or model selection.
The selected sample has no identical native-RGB duplicates or exact CIFAKE
matches under the checks described above; no candidate exclusions were needed.

| Model | CIFAKE accuracy | GenImage accuracy | GenImage precision | GenImage recall | GenImage F1 | Accuracy drop (percentage points) |
|---|---:|---:|---:|---:|---:|---:|
| CNN without augmentation | 95.745% | 58.980% | 59.877% | 54.440% | 57.029% | 36.765 |
| CNN with augmentation | 95.645% | 59.570% | 60.240% | 56.300% | 58.203% | 36.075 |
| Pretrained ConvNeXt V2-Tiny | 98.195% | 49.550% | 49.752% | 90.440% | 64.192% | 48.645 |
| Pretrained MambaVision-T | 97.430% | 49.010% | 49.495% | 97.020% | 65.550% | 48.420 |

High CIFAKE accuracy did not transfer to this external sample. The custom CNNs
had higher external accuracy, while ConvNeXt and MambaVision were close to the
50% accuracy of a constant classifier on this balanced sample. ConvNeXt
incorrectly labeled 4,567 of 5,000 real images as FAKE; MambaVision did so for
4,950. Their high FAKE recall and higher FAKE F1 must be interpreted alongside
these false-positive rates and accuracy.

Performance also varied by generator. The scratch CNNs scored about 72–75%
accuracy on SD v1.4, SD v1.5, and Wukong, but about 45–49% on ADM, BigGAN, and
VQDM. These results show limited cross-dataset robustness in this experiment;
they do not isolate generator novelty from changes in content, resolution, or
compression, and do not establish that one architecture always generalizes
better. The ImageNet-pretraining caveat above and single-seed training remain
limitations. Further model development would need separate development data;
this sample must remain held out for reporting.

Full external results, including per-generator metrics and confusion matrices:
[baseline CNN](runs/cnn-baseline/genimage_metrics.json),
[augmented CNN](runs/cnn-augmented/genimage_metrics.json),
[ConvNeXt V2](runs/convnextv2/genimage_metrics.json), and
[MambaVision-T](runs/mambavision/genimage_metrics.json).

## Checks

```bash
python -m unittest discover -s tests -v
```

Checks cover split reproducibility and leakage, learning a tiny synthetic sample,
checkpoint round trips, positive-class metrics, and augmentation restricted to
training images with labels preserved. External checks cover balanced seeded
sampling, unsafe archive paths, overlap exclusion, manifest/image integrity,
per-generator metric aggregation, reconstruction from tracked manifests, and
preservation of existing CIFAKE results. Synthetic learning checks
implementation behavior; real CIFAKE performance must come from an actual run.
For ConvNeXt and GPU checks, run the same test command using
`.venv-gpu/bin/python`. Optional ConvNeXt tests are skipped if `timm` is absent;
the CUDA training-step test is skipped when a GPU is inaccessible.
For MambaVision, use `.venv-mamba/bin/python -m unittest discover -s tests -v`.
Its integration tests need CUDA, the Mamba dependencies, and access to the pinned
model configuration/code (cached after the first model load). They compare the
CUDA scan and its gradients with the reference implementation and check a
mixed-precision training step plus checkpoint restoration. CPU-only rejection
checks run even without the optional Mamba dependencies.
