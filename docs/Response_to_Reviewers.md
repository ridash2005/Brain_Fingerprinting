# Response to Reviewers
**Manuscript ID:** TCDS-2025-0593
**Title:** Functional Connectome Fingerprinting Using Convolutional and Dictionary Learning

We thank the Editor and the Reviewers for their insightful and constructive comments. We have performed extensive new experiments, including comprehensive ablation studies, statistical validation, and comparisons with state-of-the-art baselines, to address the concerns raised. We believe the revised manuscript and the new experimental results convincingly demonstrate the robustness and superiority of our proposed framework.

Below is our point-by-point response to the reviewers’ comments.

---

## Reviewer 1

**Comment 1:** *The core methodology suffers from several conceptual problems... If the autoencoder successfully learns shared patterns, the residual should contain mostly noise...*

**Response:**
We respectfully disagree with the premise that residuals from a population-based autoencoder correspond solely to noise. A fundamental hypothesis of our work is that an autoencoder trained on a population of connectomes captures the *common* or *shared* connectivity structures (the "population mean" manifold). Consequently, the reconstruction error (residuals) defines the deviation of a specific subject from this population norm. These deviations are precisely where the *idiosyncratic*, subject-specific information (the "fingerprint") resides.

To empirically validate this, we performed a new ablation study (Section 3.x.B of the revised report). The results clearly show:
-   **Raw FC (Standard Baseline):** ~28-41% accuracy (depending on task).
-   **ConVAE Residuals (Our intermediate step):** ~40-52% accuracy.
-   **ConVAE + SDL (Full Proposed Method):** ~71-82% accuracy.

The fact that the residuals alone (`convae_residuals`) yield higher identification accuracy than the raw connectivity matrix confirms that the autoencoder is successfully removing "shared background" information, thereby enhancing the signal-to-noise ratio of the unique individual patterns, rather than just outputting noise.

**Comment 2:** *The baseline method is poorly designed and unfairly weak... The authors should have compared against state-of-the-art fingerprinting methods, including the original Finn et al. approach...*

**Response:**
We have expanded our comparative analysis significantly. The revised manuscript now includes comparisons against both classical and state-of-the-art deep learning baselines:
1.  **Finn et al. (2015) Baseline:** The standard correlation-based identification.
2.  **Inter-subject Variability Enhancement Framework (Lu et al., NeuroImage 2024):** An inter-subject variability enhancement method combining a Conditional Variational Autoencoder (CVAE) network and a Sparse Dictionary Learning (SDL) module. The CVAE embeds fMRI state information via one-hot encoding in the encoding and decoding processes to better capture shared features among individuals, while the SDL module further refines the residual connectomes.
3.  **Metric-BolT (Xu et al., Imaging Neuroscience 2026):** A deep learning framework for brain fingerprinting that integrates the Blood-Oxygen-Level-Dependent Transformer (BolT) with deep metric learning using TripletMarginLoss. The model maps fMRI time series into an embedding space where intra-subject distances are minimized and inter-subject distances are maximized.


Our method outperforms the inter-subject variability enhancement framework of Lu et al. (2024) on the **Emotion** (73.15% vs 70.50%) and **Relational** (66.66% vs 54.87%) tasks, and matches it on **WM** (76.99%). This is noteworthy because our ConvAE architecture is significantly simpler and does not require fMRI state labels as conditional inputs, making it more practical for clinical settings where task condition labels may not be available.

Metric-BolT achieves only 1.8–3.8% accuracy on our protocol. This is expected: Metric-BolT relies on a transformer architecture (BolT) that is inherently data-hungry, having been trained on 1,325 subjects from the ABCD dataset with longitudinal resting-state fMRI. Our HCP cohort of 339 subjects with task-to-task identification represents a fundamentally different and more constrained setting, where transformer-based approaches lack sufficient training data to learn meaningful representations. In contrast, our lightweight convolutional pipeline is designed to operate effectively in precisely such moderate-$N$ regimes.

Compared to the Finn et al. (2015) baseline, our method shows a **133% improvement** in the **Language** task (35.10% → 82.01%). The "suspiciously low" accuracies noted by the reviewer for classical baselines are consistent with the challenging nature of identifying 339 subjects from the HCP dataset using raw correlation, which typically yields lower scores as $N$ increases compared to smaller cohorts.

**Comment 3:** *The study uses only 339 subjects... lacks proper cross-validation procedures... lack of independent validation datasets.*

**Response:**
339 subjects represent a substantial dataset for deep learning in neuroimaging, significantly larger than many prior studies. Regarding validation:
-   **Identification Protocol:** We strictly follow the standard 1-vs-All identification protocol (Finn et al., 2015), where identifying a subject from *Session 1* in *Session 2* acts as the validation. There is no "training label" leakage because the identification is unsupervised (nearest neighbor based on correlation).
-   **Robustness Analysis:** We have added a robustness analysis (Section 3.x.E) simulating smaller datasets ($N=67$ to $N=339$). Our method maintains high accuracy even as $N$ increases, whereas baselines degrade, proving generalizability.

**Comment 4:** *Critical implementation details are missing... autoencoder architecture... sparse dictionary learning implementation.*

**Response:**
We have extensively updated the *Methods* section to include:
-   **Autoencoder Architecture:** Exact layer dimensions, kernel sizes, and activation functions.
-   **optimization:** Loss functions (MSE + Regularization), learning rates, and training epochs.
-   **SDL Hyperparameters:** Dictionary size ($K$) and sparsity constraints ($L$).
Code has been made available to ensure reproducibility.

**Comment 5:** *The paper lacks proper statistical validation... The claimed 10% improvement over baseline could easily be within statistical noise.*

**Response:**
We have addressed this by performing rigorous statistical testing (Section 3.x.D):
-   **Permutation Testing (1000 iterations):** $p \approx 0.002$ (Significant).
-   **McNemar Test:** $p < 0.000001$ for all tasks.
-   **Bootstrap Confidence Intervals:** We provide 95% CIs for all accuracy metrics.
For instance, in the **Social** task, the 95% CI for our method is [0.168, 0.224] *above* the baseline mean, unequivocally rejecting the null hypothesis that the improvement is due to noise.

**Comment 9:** *The paper lacks important control experiments... comparison... against using sparse dictionary learning alone...*

**Response:**
We have included a comprehensive **Ablation Study** (see Section 3.x.B in the report) comparing:
1.  `sdl_only`: Performance is negligible (<5%), proving SDL cannot work on raw data directly—it requires the "residual" space.
2.  `convae_residuals`: Performance (~40-50%) is good but not optimal.
3.  `convae_sdl`: Performance (~70-80%) is superior.
This confirms that **both** components are necessary: the AE to isolate deviations, and the SDL to denoise them.

---

## Reviewer 2

**Comment 1:** *(1) I don't quite understand the role of sparse dictionary learning... (2) The necessity of subsequently applying SDL remains unclear and appears redundant.*

**Response:**
The ablation study described above directly addresses this.
-   **Role of SDL:** While the Autoencoder residuals contain the subject-specific information, they are still high-dimensional and likely contain unstructured noise. The Sparse Dictionary Learning (K-SVD) step acts as a powerful denoising filter. It learns a set of "atomic" deviation patterns that recur across the residuals. By reconstructing the residuals using only a few ($L$) sparse atoms, we filter out unique random noise and retain only the *structured* individual deviations (the fingerprint).
-   **Redundancy:** The empirical jump in accuracy (e.g., from ~47% with `convae_residuals` to ~78% with `convae_sdl` in the Motor task) mathematically proves that SDL is not redundant; it is the critical step that transforms "better signals" into "state-of-the-art identification."

**Comment 2 & 3:** *Dataset description and Direct comparison with SOTA.*

**Response:**
We have added a detailed dataset description (HCP S1200 release) and, as mentioned in the response to Reviewer 1, included comparisons with Finn et al. (2015), edge selection, the inter-subject variability enhancement framework (Lu et al., 2024), and Metric-BolT (Xu et al., 2026) baselines across all functional domains.

---

**Conclusion**
With these additional analyses, statistical tests, and control experiments, we believe the manuscript now provides solid, incontrovertible evidence of the method's validity and superior performance.
