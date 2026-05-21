# Cold-Protein Sequence Similarity Audit

This report uses nearest-neighbor k-mer Jaccard similarity between each unique cold-protein test sequence and the unique train/validation protein sequences in the same dataset. Exact overlap tests whether the split is entity-cold. Jaccard values describe similarity of the frozen split and should not be interpreted as a sequence-cluster holdout.

## BioSnap cold_protein

- unique train+val proteins: 1745
- unique test proteins: 436
- exact test protein overlap with train+val: 0
- k-mer size: 5
- nearest-neighbor Jaccard mean / median: 0.078 / 0.011
- p90 / p95 / max: 0.260 / 0.392 / 0.870
- percent >= 0.3 / 0.5 / 0.7 / 0.9: 8.0% / 3.0% / 0.7% / 0.0%

| Rank | Test length | Nearest train/val length | Nearest Jaccard | Test prefix | Nearest prefix |
|---:|---:|---:|---:|---|---|
| 1 | 444 | 445 | 0.870 | `MREIVHLQAGQCGNQIGAKFWEVI` | `MREIVHLQAGQCGNQIGAKFWEVI` |
| 2 | 445 | 445 | 0.789 | `MREIVHIQAGQCGNQIGAKFWEVI` | `MREIVHLQAGQCGNQIGAKFWEVI` |
| 3 | 445 | 445 | 0.764 | `MREIVHIQAGQCGNQIGAKFWEVI` | `MREIVHLQAGQCGNQIGAKFWEVI` |
| 4 | 558 | 558 | 0.686 | `MYRYLGEALLLSRAGPAALGSASA` | `MYRYLAKALLPSRAGPAALGSAAN` |
| 5 | 222 | 222 | 0.683 | `MAEKPKLHYFNARGRMESTRWLLA` | `MAEKPKLHYSNIRGRMESIRWLLA` |
| 6 | 534 | 534 | 0.623 | `MARGLQVPLPRLATGLLLLLSVQP` | `MATGLQVPLPWLATGLLLLLSVQP` |
| 7 | 351 | 351 | 0.610 | `MGNAATAKKGSEVESVKEFLAKAK` | `MGNAAAAKKGSEQESVKEFLAKAK` |
| 8 | 375 | 375 | 0.603 | `MSTAGKVIKCKAAVLWELKKPFSI` | `MSTAGKVIKCKAAVLWELKKPFSI` |
| 9 | 450 | 444 | 0.602 | `MREIVHIQAGQCGNQIGAKFWEVI` | `MREIVHIQAGQCGNQIGAKFWEVI` |
| 10 | 494 | 494 | 0.586 | `MLASGMLLVALLVCLTVMVLMSVW` | `MLASGLLLVTLLACLTVMVLMSVW` |

## DrugBank cold_protein

- unique train+val proteins: 3404
- unique test proteins: 850
- exact test protein overlap with train+val: 0
- k-mer size: 5
- nearest-neighbor Jaccard mean / median: 0.077 / 0.011
- p90 / p95 / max: 0.219 / 0.387 / 0.987
- percent >= 0.3 / 0.5 / 0.7 / 0.9: 7.6% / 3.5% / 2.6% / 1.5%

| Rank | Test length | Nearest train/val length | Nearest Jaccard | Test prefix | Nearest prefix |
|---:|---:|---:|---:|---|---|
| 1 | 750 | 750 | 0.987 | `MKWTKRVIRYATKNRKSPAENRRR` | `MKWTKRVIRYATKNRKSPAENRRR` |
| 2 | 504 | 504 | 0.980 | `MLYFSLFWAARPLQRCGQLVRMAI` | `MLYFSLFWAARPLQRCGQLVRMAI` |
| 3 | 432 | 432 | 0.977 | `MGNNVVVLGTQWGDEGKGKIVDLL` | `MGNNVVVLGTQWGDEGKGKIVDLL` |
| 4 | 480 | 480 | 0.975 | `MNASEFRRRGKEMVDYMANYMEGI` | `MNASEFRRRGKEMVDYVANYMEGI` |
| 5 | 941 | 941 | 0.968 | `MGQACGHSILCRSQQYPAARPAEP` | `MGQACGHSILCRSQQYPAARPAEP` |
| 6 | 594 | 594 | 0.967 | `MASSTPSSSATSSNAGADPNTTNL` | `MASSTPSSSATSSNAGADPNTTNL` |
| 7 | 291 | 291 | 0.966 | `MVTKRVQRMMFAAAACIPLLLGSA` | `MVTKRVQRMMFAAAACIPLLLGSA` |
| 8 | 1235 | 1235 | 0.952 | `MFSGGGGPLSPGGKSAARAASGFF` | `MFSGGGGPLSPGGKSAARAASGFF` |
| 9 | 865 | 843 | 0.952 | `IPAFACAAAFLLHLFSSASAGAMA` | `MAKPLTDSEKRKQISVRGLAGLGD` |
| 10 | 388 | 388 | 0.949 | `MTIGIDKINFYVPKYYVDMAKLAE` | `MTIGIDKINFYVPKYYVDMAKLAE` |
