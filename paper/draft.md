# Few-Shot Day-to-Night Translation: Cycle Alignment vs. Example Imitation

## Abstract

Image-to-image translation between day and night driving scenes has important applications in autonomous driving and simulation, yet collecting paired day-night data at scale remains challenging. This paper investigates an extreme few-shot regime with a main grid of 10, 20, and 50 training examples and a post-hoc exploratory extension to 5 shots. We compare two one-step conditional models built on the same SD-Turbo backbone: **pix2pix-turbo**, which learns by direct imitation from paired data, and **CycleGAN-Turbo**, which learns by cycle-consistent alignment from unpaired domains. The models use matched LoRA ranks but retain model-specific adaptation paths and objectives. Across 3 random seeds per shot level, we evaluate SSIM, LPIPS, CLIP similarity, and CMMD on the same 200 held-out image pairs. pix2pix-turbo consistently obtains lower LPIPS, whereas CycleGAN-Turbo obtains lower CMMD, revealing a trade-off between perceptual fidelity and distributional alignment. We confirm a 5-shot breaking point for pix2pix-turbo on LPIPS (Holm-adjusted $p=0.005$). CycleGAN-Turbo shows an exploratory onset of CLIP-similarity instability at 10 shots, but this result is not significant after Holm correction. These findings provide metric-specific guidance for choosing between paired and unpaired learning paradigms under extreme data scarcity.

---

## 1. Introduction

Image-to-image translation—the task of transforming an input image from one domain to another while preserving its underlying structure—has seen remarkable progress with deep learning. Day-to-night translation of driving scenes is particularly valuable: it enables autonomous driving systems trained on daytime data to generalize to nighttime conditions, facilitates the creation of diverse simulation environments, and reduces the need for costly nighttime data collection.

Early approaches to image translation relied on paired training data, where corresponding input-output examples are required. The seminal **pix2pix** framework [Isola et al., 2017] demonstrated impressive results when such pairs are available. However, collecting perfectly aligned day-night image pairs in driving scenes is expensive and often impractical due to changing lighting, weather, and traffic conditions. This motivated the development of unpaired methods such as **CycleGAN** [Zhu et al., 2017], which learns domain mappings using cycle-consistency without requiring paired examples.

Recent advances have pushed the frontier further. The **img2img-turbo** framework [Parmar et al., 2024] introduces a general method for adapting single-step diffusion models like SD-Turbo to new tasks through adversarial learning. It offers two variants: **pix2pix-turbo** for paired settings and **CycleGAN-Turbo** for unpaired settings. They share the same pretrained backbone and parameter-efficient adaptation strategy, but use model-specific generator paths and fundamentally different learning paradigms—direct imitation versus cycle alignment.

While these models have shown strong performance with abundant data, their behavior under extreme data scarcity is not well characterized. In many real-world applications, only a handful of training examples are available. This raises critical questions: Which learning paradigm is more sample-efficient when data is scarce? At what point does generation quality become statistically unstable?

In this work, we systematically investigate these questions by comparing pix2pix-turbo and CycleGAN-Turbo across 10, 20, and 50 training pairs—with an exploratory extension to 5 shots. Our contributions are threefold:

1. We provide a controlled comparison of paired versus unpaired one-step diffusion models under extreme few-shot conditions for day-to-night translation.

2. We define and apply an explicit breaking-point criterion requiring both significant mean degradation and increased variability to identify shot levels where each paradigm becomes unstable.

3. We provide practical guidance for practitioners choosing between paired and unpaired learning under data scarcity.

---

## 2. Preliminaries and Problem Formulation

### 2.1 Problem Definition

The day-to-night image translation task can be formalized as follows. Let $\mathcal{D}$ denote the domain of daytime driving scene images and $\mathcal{N}$ the domain of nighttime images. Given an input image $x \in \mathcal{D}$, we seek a mapping $G: \mathcal{D} \rightarrow \mathcal{N}$ that produces a realistic nighttime counterpart $y = G(x)$ while preserving the scene structure—vehicles, roads, signs, and spatial layout.

In the **paired setting**, we have access to a dataset $\{(x_i, y_i)\}_{i=1}^N$ where $x_i \in \mathcal{D}$ and $y_i \in \mathcal{N}$ are aligned pairs of the same scene captured at different times. The model learns to directly replicate this transformation.

In the **unpaired setting**, we have separate collections $\{x_i\}_{i=1}^M \subset \mathcal{D}$ and $\{y_j\}_{j=1}^K \subset \mathcal{N}$ with no correspondence between individual images. The model must learn the domain mapping without explicit pair supervision.

This work investigates the **extreme few-shot regime**, where the number of available training examples is severely limited ($N \in \{5, 10, 20, 50\}$). We ask which learning paradigm—paired example imitation or unpaired cycle alignment—enables more effective learning under such constraints.

### 2.2 Background Concepts

**Stable Diffusion Turbo (SD-Turbo)** is a distilled version of Stable Diffusion that achieves high-quality image generation in a single inference step, dramatically reducing the computational cost compared to multi-step diffusion models. It serves as the backbone for both models in this study.

**Low-Rank Adaptation (LoRA)** is a parameter-efficient fine-tuning technique that injects trainable low-rank matrices into pre-trained models, enabling task adaptation with minimal additional parameters.

**Cycle-Consistency** is a core concept in unpaired image translation. A mapping $G: \mathcal{D} \rightarrow \mathcal{N}$ and its inverse $F: \mathcal{N} \rightarrow \mathcal{D}$ are trained such that $F(G(x)) \approx x$ and $G(F(y)) \approx y$, ensuring that translations preserve domain-invariant structure.

---

## 3. Design

### 3.1 Shared Backbone and Parameter-Efficient Adaptation

Both pix2pix-turbo and CycleGAN-Turbo use SD-Turbo as their pretrained backbone and adapt its VAE and U-Net with a small set of trainable parameters. Their directional generator paths are not identical, but the following components are matched across the comparison:

- **LoRA adapters** with U-Net rank 32 and VAE rank 4 for parameter-efficient fine-tuning.
- **Skip Connections** between the encoder and decoder to preserve high-frequency details and scene structure.
- **Learned VAE skip convolutions**, initialized near zero, that connect encoder features to the decoder.
- **A trainable U-Net input convolution** (`conv_in`) that adapts input processing to the image-conditioning signal.

This setup controls the backbone, LoRA capacity, resolution, optimizer, learning rate, batch size, and stopping budget while preserving the model-specific components required by the paired and unpaired objectives.

### 3.2 pix2pix-turbo: Example Imitation Paradigm

pix2pix-turbo follows a direct supervision approach. Given paired training examples $(x_i, y_i)$, the generator $G$ is trained to minimize the perceptual and structural difference between its output $G(x_i)$ and the ground-truth target $y_i$. The learning objective combines:

- **L2 loss** for pixel-level reconstruction.
- **LPIPS loss** [Zhang et al., 2018] for perceptual similarity.
- **CLIP image-text loss** between the generated output and the fixed nighttime target prompt for semantic alignment.
- **GAN loss** with a CLIP-based vision-aided discriminator to encourage realism.

This paradigm directly imitates the example transformations, leveraging the explicit correspondence between inputs and targets.

### 3.3 CycleGAN-Turbo: Cycle Alignment Paradigm

CycleGAN-Turbo adopts an unpaired learning objective inspired by CycleGAN. Without paired supervision, it learns the day-to-night mapping $G: \mathcal{D} \rightarrow \mathcal{N}$ and the inverse mapping $F: \mathcal{N} \rightarrow \mathcal{D}$ simultaneously. The objective combines:

- **Cycle consistency loss**: $L_{\text{cyc}} = \mathbb{E}_{x \sim \mathcal{D}}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim \mathcal{N}}[\|G(F(y)) - y\|_1]$, encouraging the mappings to be inverses.
- **Identity loss**: $L_{\text{id}} = \mathbb{E}_{x \sim \mathcal{D}}[\|F(x) - x\|] + \mathbb{E}_{y \sim \mathcal{N}}[\|G(y) - y\|]$, preserving domain-invariant content.
- **LPIPS loss** for perceptual consistency.
- **GAN losses** with two CLIP-based discriminators, one for each domain.

Although the day and night images are stored in filename-aligned directories, the CycleGAN-Turbo loader samples the two domains independently and therefore does not use those correspondences during training. This paradigm learns domain-level alignment without paired supervision, making it applicable when only separate collections of day and night images are available.

---

## 4. Methodology

### 4.1 Training Protocol

Both models were trained with the following shared hyperparameters:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | $1 \times 10^{-5}$ |
| Batch size | 1 |
| Resolution | $512 \times 512$ |
| Training steps | 2,000 per run |
| LoRA rank (U-Net) | 32 |
| LoRA rank (VAE) | 4 |
| Seeds per shot level | 3 |

The fixed 2,000-step budget provides a matched stopping criterion across models and shot levels. It does not imply equal computation or an equal number of epochs: the two objectives perform different operations within each global step, and smaller shot levels revisit each training image more frequently.

Model-specific runtime settings were retained for memory stability. pix2pix-turbo used full precision, 4 data-loader workers, and resized center crops at training time. CycleGAN-Turbo used FP16, gradient checkpointing, 0 data-loader workers, and direct resizing to $512 \times 512$. Both used direct resizing for held-out generation.

The pix2pix-turbo objective weights were 1.0 for L2, 5.0 for LPIPS, 5.0 for CLIP image-text loss, and 0.5 for GAN loss. The CycleGAN-Turbo weights were 1.0 for cycle L1, 10.0 for cycle LPIPS, 1.0 for identity L1, 1.0 for identity LPIPS, and 0.5 for GAN loss.

### 4.2 Dataset

We used the **DarkDriving Dataset** [7]. The raw release contained 5,906 aligned training pairs and 3,632 aligned held-out test pairs. Before any random sampling, both images in every pair were fully decoded; 6 corrupted training pairs and 2 corrupted test pairs were excluded, leaving 5,900 valid training pairs and 3,630 valid held-out test pairs. We reserved 100 valid training pairs for internal validation and sampled each few-shot training split from the remaining pool with the corresponding experiment seed. Formal evaluation used a fixed 200-pair subset drawn once from the held-out test split with sampling seed 42 and reused it for every model, shot level, and training seed.

### 4.3 Evaluation Metrics

We employ four complementary metrics to assess translation quality:

| Metric | Intuitive Question | Level | Direction |
|--------|-------------------|-------|-----------|
| **SSIM** [8] | Does the generated image preserve the target's local structure? | Per-image | ↑ better |
| **LPIPS** [4] | Is it perceptually close to the paired night-time target? | Per-image | ↓ better |
| **CLIP similarity** [9] | Are the generated and target images semantically similar? | Per-image | ↑ better |
| **CMMD** [10] | Does the generated set match the distribution of real nighttime images? | Dataset-level | ↓ better |

### 4.4 Breaking-Point Definition

We define a formal breaking point as a shot level $L$ (relative to the next higher level $H$) where two conditions hold:

1. **Significant mean degradation**: Performance at $L$ is worse than at $H$ in the direction of the metric (one-sided paired t-test across matching seed IDs).
2. **Increased variability**: For per-image metrics, both the between-seed variance ratio and the within-seed test-sample variance ratio at $L$ must exceed 1 relative to $H$. CMMD is computed once per generated set, so only its between-seed variance is applicable.

For Holm-adjusted significance, we control the family-wise error rate over the 12 adjacent-shot tests within each model (4 metrics $\times$ 3 adjacent transitions). Because each comparison contains only 3 seeds, raw-significant but Holm-nonsignificant results are reported as exploratory rather than confirmed breaking points.

### 4.5 Implementation

We build on the official [img2img-turbo repository](https://github.com/GaParmar/img2img-turbo). All formal runs were trained on a single NVIDIA RTX 5090. The source and target prompts were fixed as “a driving scene during the day” and “a driving scene during the night.” Checkpoints were selected at step 2,000; for CycleGAN-Turbo, the trainable U-Net `conv_in` weights were explicitly included in checkpoint saving and loading in addition to the LoRA and VAE weights.

Generation used base seed 0 and a deterministic filename-derived seed for each image, making each sample reproducible and independent of traversal order. The evaluation pipeline verified filenames against each generation manifest and computed per-image SSIM, LPIPS with an AlexNet backbone, and CLIP image-image similarity with ViT-B/32. CMMD was computed once per generated set with kernel $\sigma=1$. Generation and evaluation used the same 200 filenames and paired nighttime targets across every condition.

**Pseudo-code for training and evaluation:**

```text
test_pairs = sample(valid_held_out_pairs, n=200, seed=42)

for model in [pix2pix-turbo, CycleGAN-turbo]:
    for shot in [5, 10, 20, 50]:  # 5-shot is a post-hoc exploratory extension
        for seed in [1, 2, 3]:
            split = sample(valid_training_pool, n=shot, seed=seed)
            train_data = PairedDataset(split) if model == pix2pix-turbo \
                         else UnpairedDataset(split)
            
            # Train generator with 2,000 steps
            generator = train_model(model, train_data, steps=2000)
            
            # Generate on fixed 200-image test subset
            outputs = generator.generate(test_pairs.day, generation_seed=0)
            
            # Compute metrics
            metrics = evaluate(outputs, test_pairs.night)
            save(metrics, model, shot, seed)
```

---

## 5. Numerical Experiments

### 5.1 Experimental Setup

We conducted experiments across 10, 20, and 50 training shots, with an exploratory extension to 5 shots. For each configuration, we trained 3 independent seeds to capture variability. All evaluations used the same deterministic 200-image held-out test subset, ensuring fair comparison across models and shot levels.

### 5.2 Quantitative Results

We compare held-out performance across all four metrics (SSIM, LPIPS, CLIP similarity, and CMMD) for both models.

| Model | Shots | SSIM ↑ | LPIPS ↓ | CLIP similarity ↑ | CMMD ↓ |
|-------|------:|-------:|--------:|------------------:|-------:|
| CycleGAN-Turbo | 5 | 0.634 | 0.586 | 0.898 | 0.039 |
| CycleGAN-Turbo | 10 | 0.676 | 0.543 | 0.917 | 0.018 |
| CycleGAN-Turbo | 20 | 0.693 | 0.535 | 0.930 | 0.012 |
| CycleGAN-Turbo | 50 | 0.702 | 0.535 | 0.931 | 0.014 |
| pix2pix-turbo | 5 | 0.634 | 0.533 | 0.896 | 0.052 |
| pix2pix-turbo | 10 | 0.676 | 0.504 | 0.922 | 0.027 |
| pix2pix-turbo | 20 | 0.717 | 0.495 | 0.929 | 0.025 |
| pix2pix-turbo | 50 | 0.686 | 0.487 | 0.930 | 0.025 |

Values are means over 3 training seeds; uncertainty across seeds is analyzed separately below.

**Key observations:**

1. **Performance scaling**: Most metrics improve from 5 to 20 shots, while gains beyond 20 shots are generally small. pix2pix-turbo SSIM is non-monotonic at 50 shots, so the plateau is not uniform across every metric.

2. **Variance reduction**: The strongest low-shot instability is concentrated around the 5-to-10-shot transition, but its direction and magnitude depend on the model and metric.

3. **Model comparison**: pix2pix-turbo achieves lower LPIPS at every shot level, indicating better paired-target perceptual fidelity. CycleGAN-Turbo achieves lower CMMD at every shot level, indicating better distributional alignment. SSIM and CLIP similarity are broadly comparable at higher shot levels, so neither model dominates every metric.

### 5.3 Breaking-Point Analysis

We applied the breaking-point criterion to identify shot levels where generation quality begins to degrade significantly.

| Model | Estimated Onset | Primary Evidence | Mean Degradation | Variance Ratio | Statistical Status |
|-------|-----------------|-------------------|------------------|----------------|-------------------|
| **pix2pix-turbo** | 5 shots | LPIPS | +0.0293 (worse) | 6.89× / 1.28× | Holm-confirmed, p=0.005 |
| **CycleGAN-turbo** | 10 shots | CLIP similarity | -0.0126 (worse) | 1.40× / 1.12× | Exploratory: raw p=0.008; Holm p=0.091 |

**Confirmed breaking point**: pix2pix-turbo shows statistically significant LPIPS degradation at 5 shots, with both seed-level and sample-level variance ratios exceeding 1. Under our criterion, this provides evidence of instability in paired example imitation at the lowest tested shot level; it should not be generalized to fewer shots or other datasets without additional experiments.

**Exploratory candidate**: CycleGAN-turbo shows raw-significant CLIP similarity degradation at 10 shots, though the Holm-adjusted p-value (0.091) does not reach significance. This suggests potential instability at 10 shots that warrants further investigation with more seeds.

### 5.4 Sample-Level Variance

We also examine within-seed test-sample variance for the three per-image metrics. For pix2pix-turbo at 5 shots, sample variance ratios exceed 1 for SSIM (1.46×), LPIPS (1.28×), and CLIP similarity (1.56×). Only LPIPS and CLIP similarity also have seed-level variance ratios above 1; LPIPS alone satisfies the full Holm-confirmed criterion. CycleGAN-Turbo at 10 shots shows sample variance increases for SSIM (1.44×), LPIPS (1.28×), and CLIP similarity (1.12×), but only CLIP similarity combines this increase with worse mean performance and a seed-level variance ratio above 1. These descriptive sample ratios are supporting diagnostics, not independent significance tests.

---

## 6. Discussion

### 6.1 Research Questions

**Q1: Which paradigm enables more effective learning under extreme data scarcity?**

Under our setting, pix2pix-turbo is more data-efficient for paired-target perceptual fidelity, achieving better LPIPS at 5–10 shots. CycleGAN-Turbo, however, obtains better CMMD and therefore stronger distributional alignment. Robustness is metric-dependent, so the experiments do not support a universal ranking of the two paradigms.

**Q2: Where is the breaking point at which generation quality begins to degrade?**

- **pix2pix-turbo**: Breaks at 5 shots (Holm-confirmed on LPIPS, p=0.005), with mean degradation of +0.0293 and seed variance ratio of 6.89×.
- **CycleGAN-turbo**: Shows exploratory instability at 10 shots (raw p=0.008 on CLIP similarity, Holm p=0.091), with mean degradation of -0.0126 and seed variance ratio of 1.40×.

### 6.2 Implications

Our findings have practical implications for practitioners:

- At the tested **5-shot** level, pix2pix-turbo provides stronger LPIPS but has a confirmed LPIPS breaking point; CycleGAN-Turbo has lower CMMD but shows broader exploratory instability across the 5-to-10-shot transition.
- At **10–20 shots**, model selection should follow the target criterion: pix2pix-turbo favors paired-target perceptual fidelity, whereas CycleGAN-Turbo favors distributional alignment.
- At the tested **50-shot** level, SSIM and CLIP similarity are broadly comparable, but the LPIPS/CMMD trade-off remains. Claims about more than 50 shots require additional experiments.

### 6.3 Limitations

1. **Limited seeds**: With only 3 seeds per configuration, statistical power is limited. Our exploratory findings for CycleGAN-turbo at 10 shots would benefit from validation with more seeds.

2. **Single dataset**: Results may not generalize to other domains (e.g., medical imaging, satellite imagery) without further investigation.

3. **Fixed training steps**: We fixed training steps at 2,000; optimal training duration may differ between paradigms and shot levels.

4. **CMMD variability**: CMMD is a distribution-level metric with no per-sample variance, limiting our ability to assess sample-level instability for this metric.

---

## 7. Conclusions

This paper presented a controlled comparison of paired (pix2pix-turbo) and unpaired (CycleGAN-Turbo) one-step diffusion models for few-shot day-to-night image translation. Our key findings are:

1. **The models exhibit a metric-specific trade-off**: pix2pix-turbo consistently achieves better LPIPS, while CycleGAN-Turbo consistently achieves better CMMD. Variance is also model- and metric-dependent.

2. **Performance becomes broadly comparable on SSIM and CLIP similarity at 20–50 shots**, while the LPIPS/CMMD trade-off remains.

3. **One formal breaking point was confirmed**: pix2pix-turbo degrades at 5 shots on LPIPS. CycleGAN-Turbo shows an exploratory CLIP-similarity instability onset at 10 shots that does not survive Holm correction.

4. **Diminishing returns** generally set in beyond 20 shots, although pix2pix-turbo SSIM remains non-monotonic rather than showing a uniform plateau.

**Future directions** include: (a) investigating adaptive training schedules where the number of steps scales with data quantity; (b) exploring hybrid approaches that combine paired and unpaired objectives; (c) extending the analysis to other translation domains; and (d) developing methods to predict breaking points a priori based on dataset characteristics.

The code and evaluation framework used in this study are available in the accompanying repository.

---

## References

[1] Parmar, G., Park, T., Narasimhan, S., & Zhu, J.-Y. (2024). One-Step Image Translation with Text-to-Image Models. *arXiv preprint arXiv:2403.12036*. 

[2] Isola, P., Zhu, J.-Y., Zhou, T., & Efros, A. A. (2017). Image-to-Image Translation with Conditional Adversarial Networks. *CVPR*.

[3] Zhu, J.-Y., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. *ICCV*.

[4] Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. (2018). The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. *CVPR*.

[5] Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). Adversarial Diffusion Distillation. *arXiv preprint arXiv:2311.17042*. [SD-Turbo]

[6] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., ... & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*.

[7] Wang, W., Yang, H., Li, B., Sun, J., Zhao, X., Xu, Z., Guo, Q., Min, H., Zhang, T., & Yu, H. (2026). DarkDriving: A Real-World Day and Night Aligned Dataset for Autonomous Driving in the Dark Environment. *arXiv preprint arXiv:2603.18067*.

[8] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image Quality Assessment: From Error Visibility to Structural Similarity. *IEEE Transactions on Image Processing*.

[9] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.

[10] Jayasumana, S., Ramalingam, S., Veit, A., Glasner, D., Chakrabarti, A., & Kumar, S. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation. *CVPR*.
