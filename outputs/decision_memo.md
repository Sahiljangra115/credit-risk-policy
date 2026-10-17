# Decision memo: replace the flat approval rule with a cost-based score cutoff

To: Head of Credit. Date: 14 October 2026. Basis: Home Credit data, a test period of 76,878 applications the model never saw.

**Decision asked for.** Approve a randomised pilot of a model-based approval cutoff: decline an applicant when the predicted chance of default is above 19%, approve everyone else. Run it against current practice (approve all) for about 3 weeks of applications, then hold the decision on full rollout until the pilot loans mature.

**What we expect.**
- Approval rate falls from 100% to about 91%.
- Default rate among approved loans falls from 7.97% to 5.79%.
- Profit rises 5.7%, about 250 million in source currency over the test period.
- A blunt tightening is the wrong lever. Declining a fixed 30% on an affordability rule would have cut profit by 40%. Even when the model picks which 30% to decline, profit still falls 4.7%. The cutoff earns more because it declines only applicants whose expected loss is larger than the margin they bring.

**How sure we are.** The gain depends on two numbers we assumed and you should replace: margin of 15% of the loan and a loss of 60% of the loan on default. We re-ran the decision for margins of 5% to 20% and losses of 40% to 80%. The cutoff beat approve-all in every case, by between 50 million and 1.32 billion. The score ranks borrowers well, and its predicted default rates match observed ones to within 0.3 percentage points.

**Pilot design.** Split applicants 50/50 at random between current practice and the new cutoff. We need about 27,000 applicants per group, about 2.7 weeks at an assumed 20,000 applications a week, to detect the expected profit difference with 80% confidence at the usual 5% error rate. Using the score to reduce noise in the comparison saves only 6% of that sample. Defaults take months to show, so read approval rate and early missed payments at 3 weeks and read profit only after the loans mature.

**Fairness check.** At the cutoff, women are approved at 92.6% and men at 88.0% (ratio 0.95). Applicants in Home Credit's area ratings 1, 2 and 3 are approved at 97.1%, 91.4% and 85.3% (lowest-to-highest ratio 0.88). All groups clear the four-fifths rule, and each gap follows a real gap in default rates (men 10.0% vs women 6.9%; area 3 10.9% vs area 1 4.6%). Area rating 3 carries the largest decline burden and should be reviewed against RBI fair practices expectations before launch. Gender is not a model input. Age is, and compliance should confirm that is acceptable.

**Limits.** The dataset's application dates were added by its uploader, so we have not shown that the score stays stable over time. Outcomes exist only for loans that were granted, so behaviour of applicants who would have been declined is simulated. All profit figures replay observed outcomes; they are not results from a live test.

Sources (in `outputs/`): profit and approval `policy_comparison.csv`; assumption ranges `policy_sensitivity.csv`; pilot size `metrics.json` (`ab`); fairness `fairness.csv`; model accuracy `metrics.json` (`test_calibrated`).
