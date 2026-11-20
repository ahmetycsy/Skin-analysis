# stack_results

| variant                             |   macro_f1_mean |   macro_f1_sd_over_repeats |   accuracy |   macro_auc |
|:------------------------------------|----------------:|---------------------------:|-----------:|------------:|
| Average A1+A2 (no learning)         |          0.5194 |                     0      |     0.7424 |      0.8795 |
| Average + per-class thresholds      |          0.5022 |                     0.0079 |     0.7376 |      0.8776 |
| Average x metadata likelihood ratio |          0.5218 |                     0.0029 |     0.7443 |      0.8838 |
| LR: A1+A2 probabilities             |          0.4606 |                     0.0265 |     0.75   |      0.8848 |
| LR: A1+A2 probabilities + metadata  |          0.4586 |                     0.0083 |     0.7137 |      0.9259 |
| LR: metadata only                   |          0.2935 |                     0.0099 |     0.6298 |      0.8144 |
