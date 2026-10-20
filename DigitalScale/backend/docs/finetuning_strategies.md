

| Strategy                  | What’s Trainable                                                           | Learning-Rate Setup                                         |
| ------------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------- |
| **freeze\_features**      | • Classifier head only                                                     | • Single higher LR on head (e.g. 1e-3)                      |
| **unfreeze\_last\_block** | • Last denseblock + preceding transition + final norm<br>• Classifier head | • Higher LR on head (1e-3)<br>• Lower LR on backbone (1e-4) |
| **unfreeze\_all**         | • Entire network                                                           | • Higher LR on head (1e-3)<br>• Lower LR on backbone (1e-4) |

---

* **Discriminative Learning Rates**
  Splitting the optimizer into two param-groups ensures the new classifier head trains quickly (larger LR) while the pre-trained backbone adjusts more gently (smaller LR), reducing the risk of destroying useful ImageNet features.

* **Cosine Annealing LR Scheduler**
  Smoothly decays both LRs toward a small minimum over the full training run, which often yields better final convergence than a step schedule.

* **Weight Decay (1 × 10⁻⁴)**
  Acts as L2 regularization to prevent overfitting, especially important when fine-tuning on relatively small datasets.


* **BatchNorm Freezing Logic**
  When freezing layers, all BatchNorms are also set to eval mode to lock their running statistics, avoiding training instability in frozen blocks.
