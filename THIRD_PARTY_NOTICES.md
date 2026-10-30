# Third-party notices

videobrowser is licensed under the GNU Affero General Public License,
version 3 only. The full text is `LICENSE`. `pyproject.toml` sets
`license = "AGPL-3.0-only"`.

Version 3 only: the program specifies AGPL-3.0 and no later version.
`LICENSE` includes the FSF's "How to Apply These Terms to Your New Programs"
appendix, and that appendix shows an example notice with an "or any later
version" line. That example is the FSF's sample wording. The grant for
videobrowser is the SPDX identifier `AGPL-3.0-only`.

## Flagged packages

Aikido reported these from the ranges in `pyproject.toml`
(`ultralytics>=8.3.0`, `torch>=2.4`). A later resolve can move patch
versions. The license of each package stays the same across those ranges.

| Package | Version | How it is installed | License Aikido reported | With AGPL-3.0-only |
| --- | --- | --- | --- | --- |
| `ultralytics` | 8.4.192 | direct dependency | AGPLv3+ | Compatible |
| `ultralytics-thop` | 2.2.3 | required by `ultralytics` (`ultralytics-thop>=2.2.3`) | AGPLv3+ | Compatible |
| `ultralytics-platform` | 0.1.101 | required by `ultralytics` on Python >= 3.11 (`ultralytics-platform>=0.1.45`) | AGPL-3.0-only | Same license |
| `nvidia-nvshmem-cu13` | 3.7.2 | required by `torch==2.15.0` when `platform_system == "Linux"` | Proprietary | Accepted as a CUDA runtime |

`ultralytics` 8.4.192 publishes license text `AGPL-3.0` and the classifier
"GNU Affero General Public License v3 or later (AGPLv3+)". Source headers say
`AGPL-3.0`. `ultralytics-thop` 2.2.3 matches that. `ultralytics-platform`
0.1.101 sets `license_expression` to `AGPL-3.0-only`. AGPLv3+ and
AGPL-3.0-only combine under AGPL-3.0, which is this project's license.

`src/videobrowser/riders.py` imports `ultralytics.YOLO` for detection,
ByteTrack (`tracker="bytetrack.yaml"`), TensorRT export
(`export(format="engine")`), and CoreML export (`export(format="coreml")`).
The default weights are `yolo11n.pt`, which Ultralytics distributes under the
same AGPL terms. This repository does not import `ultralytics-thop` or
`ultralytics-platform` by name; both are installed because `ultralytics`
depends on them. `ultralytics-thop` is the FLOPs helper. `ultralytics-platform`
is the Ultralytics SDK.

## NVIDIA NVSHMEM

`nvidia-nvshmem-cu13` is a CUDA runtime library (NVSHMEM). videobrowser has no
import of it. On Linux, the PyPI wheel `torch==2.15.0` requires
`nvidia-nvshmem-cu13==3.7.2`. macOS wheels of that torch release do not
declare it. NVIDIA publishes the wheel, and PyTorch pulls it in unmodified
as a Linux install requirement. The wheel's metadata says
`License-Expression: LicenseRef-NVIDIA-Proprietary`. The `License.txt` in the
wheel is the NVIDIA Software Development Kit agreement (release date 26 July
2018) plus the CUDA Toolkit supplement.

The proprietary terms stay on that binary. They leave videobrowser's
AGPL-3.0-only license as it is. This finding is accepted.

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

## Aikido ignore reasons

| Finding | Ignore | Reason |
| --- | --- | --- |
| `ultralytics@8.4.192` AGPLv3+ | Yes | `Project is AGPL-3.0-only, compatible with ultralytics AGPLv3+.` |
| `ultralytics-thop@2.2.3` AGPLv3+ | Yes | `Transitive AGPLv3+ dependency of ultralytics; project is AGPL-3.0-only.` |
| `ultralytics-platform@0.1.101` AGPL-3.0-only | Yes | `Transitive AGPL-3.0-only dependency of ultralytics; project is AGPL-3.0-only.` |
| `nvidia-nvshmem-cu13@3.7.2` Proprietary | Yes | `Transitive NVIDIA CUDA runtime from the Linux PyTorch wheel (LicenseRef-NVIDIA-Proprietary); unmodified redistributable, not project source.` |

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
