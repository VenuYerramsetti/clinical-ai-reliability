# Clinical AI Reliability Architecture

```text
                HAM10000 Dataset
                        │
                        ▼
              Data Loading Pipeline
                        │
                        ▼
               Image Preprocessing
          (Resize, Tensor Conversion)
                        │
                        ▼
                 Train / Val / Test
                        │
                        ▼
               Transfer Learning
                ResNet18 Backbone
                        │
          ┌─────────────┴─────────────┐
          │                           │
          ▼                           ▼
 Frozen Early Layers         Fine-Tuned Layer4
                                        │
                                        ▼
                             Fully Connected Layer
                                  (7 Classes)
                                        │
                                        ▼
                            Weighted CrossEntropy
                                        │
                                        ▼
                            Optimizer + Scheduler
                                        │
                                        ▼
                             Validation Monitoring
                                        │
                                        ▼
                               Best Model Saving
                                        │
                                        ▼
                                 Test Evaluation
                                        │
          ┌─────────────────────────────┼────────────────────────────┐
          ▼                             ▼                            ▼
 Classification Report          Confusion Matrix             Grad-CAM
                                                                │
                                                                ▼
                                                    Explainable AI Visualizations
```