# Jev Evaluation Study -- Run 20260918-072318

## Test-retest reliability
- 338 pairs x 3 repeats
- overall_fit_score stdev across repeats: mean=0.80, median=0.66
- per-requirement score stdev: mean=0.42 (n=6957 requirement observations)
- recommendation full agreement rate: 93.5%

## Internal coherence
- n=338 assessments
- mean(requirement_scores) vs overall_fit_score: Spearman rho=0.667 (p=6.736e-45, n=338)
- hire (n=44, mean=83.9) vs no (n=71, mean=24.2): Mann-Whitney U p=1.247e-19
- meets_min_qualifications=True (n=67, mean=79.0) vs False (n=271, mean=43.3): Mann-Whitney U p=1.225e-33

## Criteria-design ablation (concrete vs. vague Score criteria)
- n=40 sampled pairs
- overall_fit_score confidence (the one Score-type fixed question, directly affected by the criteria change): concrete mean=0.641 vs vague mean=0.617 -- Wilcoxon p=0.4556, r=-0.050 (<0.5: concrete 28% vs vague 10%)
- unaffected_control confidence (overall_recommendation + meets_min_qualifications, criteria NOT changed by the ablation -- expected null effect): concrete mean=0.706 vs vague mean=0.702 -- Wilcoxon p=0.5236, r=0.013
- Requirement questions confidence (n_obs=873): concrete mean=0.902 vs vague mean=0.705 -- Wilcoxon p=8.583e-100, r=0.749 (<0.5: concrete 4% vs vague 24%)

## Efficiency
- Measured Jev latency (this study, n=1014 calls): mean=1183ms, median=1162ms, p95=1744ms
- Vendor-published (not independently verified against a reconstructed baseline in this study):
  - 70ms-500ms end-to-end (TypeSafe, self-reported, West Coast laptop)
  - vs. existing frontier LLMs: 3 to 329 seconds end-to-end (TypeSafe, self-reported)
  - claimed speedup: 40x-200x faster for comparable System One task intelligence (TypeSafe, self-reported)
