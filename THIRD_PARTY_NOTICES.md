# Third-party notices

videobrowser has no `LICENSE` file, and `pyproject.toml` sets no `license`
field. No project license is chosen in this repository. This file records the
dependency terms Aikido flagged on `main`, and the choice that is still open.

Versions are the ones that scan resolved from the ranges in `pyproject.toml`
(`ultralytics>=8.3.0`, `torch>=2.4`). A later resolve can move patch versions.
The license of each package stays the same across those ranges.

## Flagged packages

| Package | Version | How it is installed | License Aikido reported |
| --- | --- | --- | --- |
| `ultralytics` | 8.4.192 | direct dependency | AGPLv3+ |
| `ultralytics-thop` | 2.2.3 | required by `ultralytics` (`ultralytics-thop>=2.2.3`) | AGPLv3+ |
| `ultralytics-platform` | 0.1.101 | required by `ultralytics` on Python >= 3.11 (`ultralytics-platform>=0.1.45`) | AGPL-3.0-only |
| `nvidia-nvshmem-cu13` | 3.7.2 | required by `torch==2.15.0` when `platform_system == "Linux"` | Proprietary |

`ultralytics` 8.4.192 publishes license text `AGPL-3.0` and the classifier
"GNU Affero General Public License v3 or later (AGPLv3+)". Source headers say
`AGPL-3.0`. `ultralytics-thop` 2.2.3 matches that. `ultralytics-platform`
0.1.101 sets `license_expression` to `AGPL-3.0-only`.

`nvidia-nvshmem-cu13` 3.7.2 sets `License-Expression` to
`LicenseRef-NVIDIA-Proprietary`. The `License.txt` in the wheel is the NVIDIA
Software Development Kit agreement (release date 26 July 2018) plus the CUDA
Toolkit supplement.

## Ultralytics (one decision for three findings)

`src/videobrowser/riders.py` imports `ultralytics.YOLO` for detection,
ByteTrack (`tracker="bytetrack.yaml"`), TensorRT export
(`export(format="engine")`), and CoreML export (`export(format="coreml")`).
The default weights are `yolo11n.pt`, which Ultralytics distributes under the
same AGPL terms. This repository does not import `ultralytics-thop` or
`ultralytics-platform` by name; both are installed because `ultralytics`
depends on them. `ultralytics-thop` is the FLOPs helper. `ultralytics-platform`
is the Ultralytics SDK.

AGPLv3+ and AGPL-3.0-only can be combined. The combination is distributed
under AGPL-3.0. Declaring this project `AGPL-3.0-only` lines up with the
strictest of the three (`ultralytics-platform`). Declaring it
`AGPL-3.0-or-later` also combines; the distributed combination is still
AGPL-3.0 because of `ultralytics-platform`.

Nothing below applies a license. The options are:

1. **License videobrowser AGPL-3.0-only.** Add the AGPL-3.0 license text as
   `LICENSE` and set the `pyproject.toml` license to `AGPL-3.0-only`. The
   three packages then match the project. Anyone who distributes a modified
   copy offers the source under AGPL-3.0. Running the CLI locally on your own
   footage does not by itself require publishing changes. AGPL section 13
   (network use of a modified copy) applies when other people interact with
   that modified copy remotely. This program is a local CLI, so that clause
   matters when a modified copy is served to other people.

2. **Leave the project unlicensed and keep the dependencies.** The three
   Aikido findings stay open until they are accepted by hand. The git tree
   contains no copy of the Ultralytics packages. An installer or image that
   combines this code with those packages is a distribution of that
   combination, which calls for option 1 or option 4.

3. **Replace the detector.** Remove `ultralytics` from `pyproject.toml`. That
   drops `ultralytics-thop` and `ultralytics-platform` with it. `riders.py`
   needs a new loader in place of `YOLO.predict`, `YOLO.track`, and the
   TensorRT and CoreML export paths. The `yolo11n.pt` weights have to change
   too; they are under the same AGPL terms. This notice does not pick a
   replacement model.

4. **Take an Ultralytics enterprise license.** Ultralytics sells a commercial
   license that replaces AGPL for their packages
   (<https://ultralytics.com/license>). The three findings can then be
   accepted against that contract. This repository records no enterprise
   license.

## NVIDIA NVSHMEM

`nvidia-nvshmem-cu13` is a CUDA runtime library (NVSHMEM). videobrowser has no
import of it. On Linux, the PyPI wheel `torch==2.15.0` requires
`nvidia-nvshmem-cu13==3.7.2`. macOS wheels of that torch release do not
declare it. NVIDIA publishes the wheel, and PyTorch pulls it in unmodified
as a Linux install requirement. The proprietary terms stay on that binary
and leave the project license (still unset) alone.

The CUDA supplement in the wheel allows redistribution of the components
named in Attachment A, in object form, inside an application that has
material additional functionality. Attachment A lists the classic CUDA
runtime libraries (cudart, cuBLAS, cuFFT, and the rest of that list) and
leaves NVSHMEM unnamed. The copy this project actually installs is the PyPI
wheel PyTorch requires, under the `LicenseRef-NVIDIA-Proprietary` marker
shipped in that wheel.

The same Linux torch wheel also requires other proprietary NVIDIA wheels:
`cuda-toolkit`, `nvidia-cudnn-cu13`, `nvidia-cusparselt-cu13`, and
`nvidia-nccl-cu13`. Aikido flagged `nvidia-nvshmem-cu13` only. The same
acceptance applies to the others if a later scan lists them.

`Dockerfile.ci` installs `torch` on Linux, and the CI workflow pushes that
image to GHCR, so the image can contain these wheels. The git tree does not.

Accept this finding in Aikido. Suggested reason:

`Transitive NVIDIA CUDA runtime from the Linux PyTorch wheel (LicenseRef-NVIDIA-Proprietary); unmodified redistributable, not project source.`

## What to mark ignored in Aikido

| Finding | Ignore now? | One-line reason when you accept it |
| --- | --- | --- |
| `ultralytics@8.4.192` AGPLv3+ | After option 1 or option 4. Option 3 removes the finding. Leaving it unlicensed (option 2) is a risk acceptance, and the finding stays a real AGPL dependency. | Option 1: `Project is AGPL-3.0-only, compatible with ultralytics AGPLv3+.` Option 4: `Covered by an Ultralytics enterprise license.` |
| `ultralytics-thop@2.2.3` AGPLv3+ | Same decision as `ultralytics`. | Option 1: `Transitive AGPLv3+ dependency of ultralytics; project is AGPL-3.0-only.` |
| `ultralytics-platform@0.1.101` AGPL-3.0-only | Same decision. AGPL-3.0-only on the project covers this "-only" package. | Option 1: `Transitive AGPL-3.0-only dependency of ultralytics; project is AGPL-3.0-only.` |
| `nvidia-nvshmem-cu13@3.7.2` Proprietary | Yes. | `Transitive NVIDIA CUDA runtime from the Linux PyTorch wheel (LicenseRef-NVIDIA-Proprietary); unmodified redistributable, not project source.` |

## Other direct dependencies

These were not in the Aikido license findings. License metadata is from the
current PyPI release of each package (October 2026).

| Package | License metadata |
| --- | --- |
| `lap` | BSD-2-Clause |
| `opencv-python-headless` | Apache-2.0 |
| `numpy` | BSD-3-Clause, and the wheel expression also names 0BSD, MIT, Zlib, and CC0-1.0 for bundled pieces |
| `torch` | BSD-3-Clause project license. The wheel expression also names Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSL-1.0, ISC, MIT, and Zlib. Linux CUDA dependencies are proprietary; see above. |
| `tqdm` | MPL-2.0 AND MIT |
| `pycairo` | LGPL-2.1-only OR MPL-1.1 |
| `tzdata` | Apache-2.0 |
| `pytest` (dev extra) | MIT |
| `playwright` (`fetch` extra) | Apache-2.0 |
