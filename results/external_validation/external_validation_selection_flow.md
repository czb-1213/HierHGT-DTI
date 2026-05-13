# External Validation Selection Flow

| panel | source records | positives considered | exact mappings | GABA mapped | non-GABA mapped | held out/unresolved | role |
|---|---:|---:|---:|---:|---:|---|---|
| BioSNAP random case-study audit | 10 | 5 | 5 | 5 | 0 | 5: saved sampled-negative examples were not used for pharmacology-grounded validation | five-case GABAA alpha-subunit topology audit |
| BioSNAP cold-protein top-20 audit | 20 | 20 | 20 | 19 | 1 | 0: none | deployment-relevant family-level corroboration |
| DrugBank cold-drug top-120 breadth audit | 120 | 120 | 92 | 62 | 30 | 28: exact-sequence UniProt mapping unavailable or PubChem lookup failed | identity-distribution background and non-GABA scope check |
