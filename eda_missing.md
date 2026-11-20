# eda_missing

| column                 |   missing_lesions |   missing_pct |   missing_images |   model_input | note                                       |
|:-----------------------|------------------:|--------------:|-----------------:|--------------:|:-------------------------------------------|
| site                   |                31 |          0.59 |              nan |             1 | nan                                        |
| age_approx             |                20 |          0.38 |               40 |             1 | nan                                        |
| sex                    |                 0 |          0    |                0 |             1 | nan                                        |
| skin_tone_class        |                 0 |          0    |              nan |             1 | nan                                        |
| anatom_site_special    |              5137 |         98.03 |            10274 |             0 | only filled for acral / oral-genital sites |
| diagnosis_4            |              4479 |         85.48 |             8958 |             0 | sub-type level, often not applicable       |
| melanocytic            |              4044 |         77.18 |             8088 |             0 | only marked when True                      |
| anatom_site_general    |              1956 |         37.33 |             3912 |             0 | nan                                        |
| diagnosis_3            |                79 |          1.51 |              158 |             0 | fine-grained diagnosis                     |
| concomitant_biopsy     |                 0 |          0    |                0 |             0 | nan                                        |
| diagnosis_1            |                 0 |          0    |                0 |             0 | nan                                        |
| diagnosis_2            |                 0 |          0    |                0 |             0 | nan                                        |
| diagnosis_confirm_type |                 0 |          0    |                0 |             0 | nan                                        |
| train_meta.age_approx  |               nan |          0.38 |               40 |           nan | from --train-meta                          |
| train_meta.site        |               nan |          0.59 |               62 |           nan | from --train-meta                          |
