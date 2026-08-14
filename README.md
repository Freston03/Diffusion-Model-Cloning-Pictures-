# Generative AI and Diffusion Models: A Statistical Physics Analysis

Code accompanying my MSc thesis (Data Science and Statistical Learning, University of Limerick), reviewing the mathematical foundations of diffusion models and implementing the forward/reverse diffusion pipeline across four case studies: a synthetic 2D example, a pretrained CelebA-HQ model, a DDPM trained from scratch on MNIST, and identity-preserving fine-tuning with LoRA.

The theoretical review centres on Biroli & Mézard (2023), *"Generative diffusion in very large dimensions"*, which analyses diffusion models as high-dimensional disordered systems and derives sample-complexity thresholds ($P \gg N$, $P \gg N^2$) and a symmetry-breaking transition ($t^* = \tfrac{1}{2}\log N$) governing when a generated sample commits to a class.

---

## Contents

---

## Overview

| Case study | Dataset | Model | Trainable params |
|---|---|---|---|
| Score function visualisation | Synthetic 2D Gaussian mixture ($P=300$) | Analytic (no network) | — |
| Two-phase generation | CelebA-HQ (via pretrained model) | `google/ddpm-celebahq-256` | 0 (inference only) |
| DDPM from scratch | MNIST | Custom U-Net | ≈ 1.6M |
| Identity-preserving fine-tuning | Personal photographs (< 10) | Stable Diffusion 1.5 + LoRA ($r=16$) | ≈ 1M (< 0.2% of 860M) |

---

## Setup

````bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
````

### Requirements

````
torch
torchvision
diffusers
transformers
accelerate
peft
matplotlib
numpy
````

Recommended: run on Google Colab with a T4 GPU for the CelebA-HQ and LoRA notebooks; MNIST training runs in under an hour on CPU, faster on GPU.

---

## Usage

Each notebook is self-contained and can be run top to bottom in Google Colab or a local Jupyter environment.

````bash
jupyter notebook notebooks/01_score_function_2d.ipynb
````

**01 — Score function visualisation**
Generates a synthetic two-cluster dataset in 2D, computes the score function analytically, and visualises the symmetry-breaking transition directly as reverse trajectories splitting at the critical time.

**02 — CelebA-HQ pretrained model**
Loads `google/ddpm-celebahq-256`, runs the full 50-step reverse process, and tracks image sharpness at every step to quantify the transition between the class-selection and detail-filling phases.

**03 — MNIST from scratch**
Implements the forward process, the U-Net score network, the denoising training loop, and the reverse-time sampler entirely from first principles, following Ho et al. (2020).

**04 — LoRA fine-tuning**
Fine-tunes Stable Diffusion 1.5 on a small set of personal photographs using Low Rank Adaptation, associating a unique token with the subject's identity and generating new images in unseen scenarios.

---

## Key results

* Sample complexity: eigenvalue structure is recovered once $P \gg N$; eigenvector (directional) structure requires the far stricter $P \gg N^2$.
* Symmetry breaking: generation splits into two phases at $t^* \sim \tfrac{1}{2}\log N$ — verified both on a pretrained model and on a model trained from scratch.
* Small-data limitation: fine-tuning on fewer than 10 personal photographs is well below the $P \gg N^2$ threshold, producing convincing scene composition but inconsistent facial identity — a direct, practical demonstration of the theory.

Full derivations, figures, and discussion are in the accompanying thesis (`thesis/thesis.pdf`).

---

## References

* Anderson, B. D. O. (1982). Reverse-time diffusion equation models. *Stochastic Processes and their Applications*, 12(3), 313–326.
* Biroli, G., & Mézard, M. (2023). Generative diffusion in very large dimensions. *Journal of Statistical Mechanics: Theory and Experiment*, 2023(9), 093402.
* Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*, 33, 6840–6851.
* Hu, E. J., et al. (2021). LoRA: Low-rank adaptation of large language models. *arXiv:2106.09685*.
* Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. *CVPR*, 10684–10695.

Full bibliography in the thesis.

---

## Author

Shardul Belgamwar
MSc Data Science and Statistical Learning, University of Limerick

## AI usage disclosure

Portions of this codebase and its accompanying thesis were drafted with the assistance of a large language model (Anthropic's Claude), used to help structure and explain the implementation. All technical content, experiments, and results are my own.

## License

MIT — see `LICENSE` for details.

````

Replace `<Freston03>/<Diffusion Model(Cloning Pictures>` with your actual GitHub path, and adjust the file tree if your actual folder/notebook names differ from what I assumed. Want me to also generate a matching `requirements.txt` or `LICENSE` file?
````
