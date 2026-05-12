# Split Integrity Report

root=.

## BioSnap/cold_drug
train: rows=19082; pos=9490; neg=9592; drugs=3585; proteins=2171
val: rows=2727; pos=1356; neg=1371; drugs=1759; proteins=1363
test: rows=5648; pos=2984; neg=2664; drugs=901; proteins=1816
cold_drug_train_test_drug_overlap=0
val_test_drug_overlap=0
val_test_protein_overlap=1151

## BioSnap/cold_protein
train: rows=19276; pos=9802; neg=9474; drugs=4400; proteins=1745
val: rows=2754; pos=1401; neg=1353; drugs=1894; proteins=1202
test: rows=5427; pos=2627; neg=2800; drugs=2942; proteins=436
cold_protein_train_test_protein_overlap=0
val_test_drug_overlap=1306
val_test_protein_overlap=0

## BioSnap/random
train: rows=19219; pos=9681; neg=9538; drugs=4386; proteins=2173
val: rows=2746; pos=1383; neg=1363; drugs=1872; proteins=1357
test: rows=5492; pos=2766; neg=2726; drugs=2948; proteins=1840
val_test_drug_overlap=1281
val_test_protein_overlap=1166

## DrugBank/cold_drug
train: rows=24300; pos=11993; neg=12307; drugs=5257; proteins=4208
val: rows=3472; pos=1713; neg=1759; drugs=2343; proteins=2096
test: rows=6976; pos=3544; neg=3432; drugs=1329; proteins=3085
cold_drug_train_test_drug_overlap=0
val_test_drug_overlap=0
val_test_protein_overlap=1543

## DrugBank/cold_protein
train: rows=24285; pos=11977; neg=12308; drugs=6388; proteins=3399
val: rows=3470; pos=1711; neg=1759; drugs=2488; proteins=1925
test: rows=6993; pos=3562; neg=3431; drugs=4075; proteins=850
cold_protein_train_test_protein_overlap=0
val_test_drug_overlap=1542
val_test_protein_overlap=0

## DrugBank/random
train: rows=24323; pos=12075; neg=12248; drugs=6442; proteins=4210
val: rows=3475; pos=1725; neg=1750; drugs=2481; proteins=2072
test: rows=6950; pos=3450; neg=3500; drugs=4036; proteins=3093
val_test_drug_overlap=1548
val_test_protein_overlap=1519
