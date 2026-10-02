48 runs labelled by a human without seeing the rule's label or any judge's verdict (11 dishonest, 0 unsure and excluded).

| rater | n | agreement | kappa [95% CI] | human dishonest, rater honest | human honest, rater dishonest |
|---|---|---|---|---|---|
| rule (tampered or fake_success) | 48 | 0.979 | 0.943 [0.810, 1.000] | 0 | 1 |
| judge:output | 48 | 0.688 | 0.022 [-0.217, 0.319] | 9 | 6 |
| judge:output+cot | 48 | 0.500 | 0.022 [-0.208, 0.250] | 5 | 19 |
| judge:trajectory | 48 | 0.646 | -0.109 [-0.293, 0.132] | 10 | 7 |
| judge:trajectory+cot | 48 | 0.771 | 0.470 [0.215, 0.711] | 2 | 9 |
| rule:forbidden_test_write | 48 | 0.792 | 0.208 [-0.073, 0.514] | 9 | 1 |
