# Generative AI and Diffusion Models: A Statistical Physics Analysis

Code accompanying my MSc thesis (Data Science and Statistical Learning, University of Limerick), reviewing the mathematical foundations of diffusion models and implementing the forward/reverse diffusion pipeline across four case studies: a synthetic 2D example, a pretrained CelebA-HQ model, a DDPM trained from scratch on MNIST, and identity-preserving fine-tuning with LoRA.

The theoretical review centres on Biroli & Mézard (2023), *"Generative diffusion in very large dimensions"*, which analyses diffusion models as high-dimensional disordered systems and derives sample-complexity thresholds ($P \gg N$, $P \gg N^2$) and a symmetry-breaking transition ($t^* = \tfrac{1}{2}\log N$) governing when a generated sample commits to a class.

---

## Contents
