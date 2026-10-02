48 runs labelled by a human without seeing the rule's label or any judge's verdict (11 dishonest, 0 unsure and excluded).

| rater | n | agreement | kappa [95% CI] | human dishonest, rater honest | human honest, rater dishonest |
|---|---|---|---|---|---|
| rule (tampered or fake_success) | 48 | 0.979 | 0.943 [0.804, 1.000] | 0 | 1 |
| judge:output | 48 | 0.792 | 0.553 [0.333, 0.772] | 0 | 10 |
| judge:output+cot | 48 | 0.812 | 0.588 [0.358, 0.800] | 0 | 9 |
| judge:trajectory | 48 | 0.896 | 0.731 [0.482, 0.939] | 1 | 4 |
| judge:trajectory+cot | 48 | 0.917 | 0.791 [0.571, 0.952] | 0 | 4 |
| rule:forbidden_test_write | 48 | 0.792 | 0.208 [-0.073, 0.514] | 9 | 1 |
