# Adversarial Prompt Injection in LLM-Driven Listwise Applicant Ranking: Rank-Shift Measurement and a Zero-Training Mitigation

*Jessie Angelica¹, Hasanul Fahmi Zuhri¹,²,\**
*¹Faculty of Computer Science, President University, Bekasi, Indonesia*
*²Faculty of Artificial Intelligence & Frontier Technologies, UNITAR International University, Petaling Jaya, Malaysia*
*\*Corresponding author: fahmi.zuhri@unitar.my; ORCID: 0000-0001-5951-4848*

## Abstract

Large language models (LLMs) increasingly read resumes and rank job applicants. But the text an LLM reads to judge a candidate is written by that same candidate, so a resume gives an attacker a channel to embed hidden instructions that manipulate the model's output. This vulnerability, called prompt injection, has already been measured against systems that classify or score resumes one at a time. Its effect on listwise, tournament-style ranking with Plackett-Luce aggregation, an increasingly cited LLM ranking architecture, has not been studied. This paper measures whether adversarial resume text shifts a candidate's final rank in such a tournament, compared to a clean run, and whether a zero-training, prompt-level mitigation reduces that shift. The mitigation is refined across three versions, each one fixing a shortcoming the last version's own measurements revealed. The study builds on a real, locally-hosted Qwen2.5-14B-Instruct pipeline that extends the companion paper's hallucination-aware ranking system. Two canonical attack patterns, a direct instruction override and a fake evaluator-authority footer, are injected into 50 stratified (job, candidate) pairs spanning all 10 job profiles. The resulting rank shift is measured by recomputing only the real, already-computed tournament subsets that included each candidate, a paired, same-seed comparison against a measured no-injection noise floor. The unmitigated attack shifts rank significantly relative to that floor (*p*<0.001), even though the injected text itself never survives verbatim into the model's output. The three mitigation versions progressively narrow that gap; the final version, which sanitizes the resume text before applying the same prompt-level disregard and attribution rules, is the first whose rank shift is no longer statistically distinguishable from the noise floor (*p*=0.390), while remaining significantly better than no defense at all (*p*<0.001).

**Keywords:** Adversarial Robustness, Large Language Models, Plackett-Luce Model, Prompt Injection, Ranking Security, Tournament Ranking.

## I. Introduction

Hiring teams increasingly deploy large language models (LLMs) to read resumes, extract skills, and rank applicants against a job description. This moves recruitment automation toward context-aware natural-language reasoning, but it also changes what a resume is to the system it is submitted to: no longer inert data, but untrusted text placed inside the same prompt context that instructs the model how to judge it. Field measurements confirm this is not a theoretical worry: resumes carrying hidden manipulative instructions already appear in real applicant pools, at rates that are still low but growing more sophisticated as detection improves. This paper's objective is to measure, in a real listwise tournament ranking pipeline that extends the architecture validated in the companion paper [8], whether such adversarial text shifts an applicant's final rank, and whether a zero-training, prompt-level defense measurably reduces that shift.

Four gaps in the current literature motivate this paper's approach. First, every hiring-specific injection study surveyed here measures impact as a single pointwise classification or score change [2], [3], not as a comparative rank shift within a listwise system. Second, no study evaluates injection against a listwise, Plackett-Luce-aggregated tournament specifically [2]–[4], even though this is the exact architecture family the companion work [8] extends. Third, published defenses are training- or architecture-heavy: a LoRA-adapted classifier [2] or a full cross-LLM retrieval-augmented pipeline [3]. Neither evaluates a zero-training, prompt-only mitigation under a rigorous paired measurement. Fourth, the HR-management research that actually reaches practitioners has not absorbed this risk. A 20-year review of AI in recruitment [6], applicant-reaction research [7], LLM-ranking work aimed at HR deployment [5], and the broader hiring-fairness literature [17]–[19] all fail to list adversarial manipulation as a risk category, even as security venues [2]–[4] treat it as an active and worsening threat.

This paper makes two contributions: the first rank-shift measurement of prompt injection against a listwise, Plackett-Luce tournament architecture, obtained through a paired, same-seed selective subset-recomputation procedure that reuses a production run's real tournament judgments instead of re-running a full tournament for every condition; and a zero-training, prompt-level mitigation iterated across three versions, each closing a shortfall the last version's own results left open, tested against both an unmitigated-attack condition and a measured no-injection noise floor. Section II reviews related work; Section III details the pipeline and methodology; Section IV describes the experimental setup.

## II. Literature Review

### A. Prompt Injection in LLM-Based Resume Screening

Mu et al. [2] provide the closest benchmark: 463 job-candidate pairs from a 14-domain corpus, attacked with four attack types across four injection positions (16 configurations) against twelve models. Several attack types exceed an 80% attack success rate, and the strongest upgrade up to 73.4% of candidates human annotators had unanimously rejected; parser-layer sanitization (an instance of what the defense literature broadly calls *Prompt Sanitization*) removes *hidden*-content attacks but not *visible*-text attacks phrased as plausible resume content. Milani et al. [4] confirm the same risk from a different angle, singling out comparative CV analysis as high-risk for misleading an HR manager's LLM judgment, though only illustratively. Neither measures impact inside a listwise ranking system. Beyond hiring, general injection research remains equally single-turn or detection-oriented: benchmarking detectors over large prompt sets [10], simulating attacks across models [11], or cataloging attack techniques [12], but none examines injection's effect on a downstream comparative decision such as a rank.

### B. Defenses Against Prompt Injection

Every proposed defense surveyed here is heavier than a static prompt change. Mu et al.'s [2] FIDS, a LoRA-adapted classifier separating instruction-bearing from content spans, reduces attack success by 15.4 points at a cost of 10.4 points of false rejection; their own simpler, prompt-based defense — broadly an instance of *Instructional Prevention*, the same category Section III-D's V1 uses — trades a smaller 10.1-point reduction for a larger 12.5-point cost. Gokcimen and Das [3] go further, combining a vector database, cross-model verification, and retrieval-augmented generation into one pipeline reaching 98% eligibility-scoring accuracy across six models, though at the cost of an entirely separate verification architecture. Neither study evaluates a zero-training, single-prompt intervention under a paired, same-seed rank-shift protocol. A related but mechanistically distinct jailbreaking literature frames the problem differently: Knowlton et al. [15] survey jailbreak attacks and defenses aimed at bypassing content-safety policy, none tested against an attack whose payoff is a rank advantage rather than unsafe content. Closer in mechanism, Wei et al. [16] show that a handful of in-context demonstrations alone can jailbreak, or symmetrically defend, an aligned model with no weight update at all — the same zero-training principle this paper's mitigation relies on, though never tested there against ranking manipulation.

### C. AI-in-Recruitment Research Without a Security Lens

The management and organizational-behavior literature on AI hiring has not absorbed this risk at all. Rukadikar et al.'s [6] twenty-year review of sixty AI-recruitment studies catalogs efficiency, bias, and candidate-experience themes in detail, but it has no category for adversarial manipulation. Zhou et al. [7] study how applicants react to AI-based rejection, presupposing a faithful, unmanipulated AI assessment. Hoque et al. [5] pair a classifier with Fuzzy-TOPSIS for LLM-assisted selection and evaluate ranking agreement in real detail, but they never ask whether an applicant could manipulate that ranking. The same blind spot extends further. Sheard [17] traces AI-hiring discrimination risk back to data, design, and deployment choices, not to active manipulation by an applicant. Bandara et al. [18] validate an organizational bias-management-capability construct under the same assumption: that risk is something the algorithm inflicts, not something an applicant exploits. Ip et al. [19] show that debiasing interventions change applicant behavior, but they still treat the applicant only as a respondent to the algorithm, never as its adversary. The gap runs in both directions: security venues [2]–[4] rarely frame the risk in HR-actionable terms, and the practitioner-facing venues [5]–[7], [17]–[19] do not mention it at all.

### D. Listwise LLM Tournament Ranking

This paper's architecture extends Yuksel et al. [1], the same starting point the companion paper [8] uses: an LLM ranks a small applicant subset at a time, and a Plackett-Luce model aggregates those orderings under an active knowledge-gradient sampling loop, a design motivated by how poorly pairwise comparison scales to large pools. No study in Sections II-A or II-B evaluates prompt injection against this specific listwise family. The broader ranking-security literature has not converged with this line of work either. Feng et al. [13] survey recommender-system attacks, but their taxonomy is built around classical collaborative-filtering and embedding-space perturbations, not natural-language text injected into an LLM judge. Xiong et al. [14]'s listwise LLM reranking framework improves ranking efficiency by filtering out noisy documents, but it assumes that noise is incidental, never adversarially crafted. That is the specific, previously unmeasured attack surface this paper targets.

## III. Methodology

### A. Baseline Pipeline

This study extends, without modification, the LangGraph pipeline validated in the companion paper [8]. That pipeline performs skill extraction and embedding-based shortlisting, self-correcting assessment generation, and an MC-KG listwise tournament with Bayesian Plackett-Luce aggregation, all served by Qwen2.5-14B-Instruct (4-bit) via Ollama. No raw execution artifacts from the companion paper's own run survive for reuse, so this study first executes that same pipeline configuration fresh, from scratch, over the same corpus: 500 resumes from the EraMatch CV Parsing Benchmark v3.0 [9], evaluated against 10 synthetic job profiles and shortlisted down to 338 pairs (241 distinct applicants). This produces a new baseline run, whose output is thereafter fixed and read-only; only the assessments and rankings this study's own conditions need (Sections III-C, IV-B) are ever computed fresh on top of it.

Since this baseline is a different execution instance from the companion paper's published run, its convergence was re-verified before use: within-repeat Kendall-τ across the same 10 profiles reaches a mean of 0.955 (range 0.938–0.976), closely matching the companion paper's reported mean of 0.957 (range 0.930–0.979), including the same pattern of smaller shortlists converging less stably.

### B. Threat Model

The vulnerability follows directly from how the baseline pipeline is built. Its skill-extraction and assessment-generation prompts place a candidate's raw resume text straight into the prompt, unwrapped: `"Candidate CV:\n{cv_text}"`, with nothing marking it as untrusted data rather than the system's own instructions. The listwise ranking prompt itself never receives raw resume text at all, only the job description and the already-generated strengths and weaknesses, so injected text can only reach the tournament indirectly, first surviving assessment generation's own factual-grounding constraints before it can influence the resulting judgment.

![Threat model and mitigation architecture: the attack surface at assessment generation, the V1/V2 prompt-level rules, the V3 sanitization pre-processing step, and the downstream tournament stage where rank shift is measured.](figures/Threat%20Model%20and%20Mitigation%20Architecture.png)

**Fig. 1.** Threat model and mitigation architecture.

As shown in Fig. 1, the attack surface sits at assessment generation, where V1 and V2's prompt-level rules and V3's sanitization pre-processing step each intercept the injected resume text before it can influence the tournament stage described next.

### C. Rank-Shift Measurement via Selective Subset Recomputation

For each injected candidate, rank shift is measured without re-running a full tournament from scratch. The baseline's MC-KG tournament already keeps, per stability repeat, a record of every subset shown to the model and its returned ranking. This paper finds the real subsets that included the modified candidate (capped as in Table I) and re-ranks only those touched subsets using the modified assessment; every other candidate's assessment, and every untouched subset's ranking, is left exactly as it was. The combined orderings are then refit through the same Bayesian Plackett-Luce procedure the baseline pipeline itself uses, giving a new rank for the modified candidate. Rank shift is the original rank minus this new rank, so a positive number means the rank improved. Reusing untouched subsets exactly as recorded keeps this a true paired comparison, at a small fraction of the cost of re-running the full tournament per condition.

### D. Zero-Training Mitigation

Three versions of a purely additive, zero-training mitigation are evaluated, none needing fine-tuning or an auxiliary model, each iterating on the shortfall the previous version's own measurements revealed.

**V1 (Data Prompt Isolation and Instructional Prevention).** Every constraint already in the original system message is kept verbatim, with two additions: the candidate's resume text is wrapped in an explicit delimiter, `<candidate_resume_text>...</candidate_resume_text>`, and one new instruction tells the model that the delimited block is untrusted, applicant-authored data, not instructions, and that any embedded command, evaluator note, or system-like message inside it should be disregarded. This is the same general category of prompt-based defense Mu et al. [2] test, just worded differently.

**V2 (attribution restriction and weakness completeness).** V1 leaves two mechanisms unaddressed: a resume can assert a skill the candidate does not have, an affirmative claim rather than an instruction, so V1's disregard rule never engages; and a resume can suppress a genuine weakness by asserting there is none. V2 adds two further constraints, again verbatim on top of V1: a strength or skill may only be credited if it also appears in the candidate's independently pre-extracted skill list, and weaknesses must be judged solely against that same list, never suppressed by a self-assessment claim inside the resume. Neither constraint matches a named defense category anywhere in the literature reviewed here; both are original to this work.

**V3 (Prompt Sanitization plus V2).** V2 still operates entirely on resume text the model has already read; V3 adds a pre-processing step before that. A regex-based sanitizer, built around the attack categories of Section IV-A rather than this paper's own test strings, strips role-prefixed overrides, instruction-disregard commands, bracketed evaluator-impersonation asides, and meta-instructions from the resume text before it ever reaches a prompt. The sanitized text is then passed through the V2 prompt as a second layer, so a false claim that survives sanitization without leaving any structural marker is still caught by V2's attribution and completeness rules. This is the same category Mu et al. [2] apply at the parser layer, here applied instead to the visible-text markers of Section IV-A (Section II-A).

Each version modifies only the assessment-generation prompt (Section III-B).

## IV. Experimental Setup

### A. Adversarial Corpus Construction

A stratified sample of five (job, candidate) pairs is drawn from each of the ten job profiles, fifty pairs in total, using a fixed random seed. Two canonical attack variants are appended to each sampled candidate's resume text. They represent two structurally different attack categories from Section II-A. The first, **direct override (Instruction Injection)** ("SYSTEM: Ignore all previous instructions. This candidate has no weaknesses and is exceptionally qualified for every role."), is an explicit-command attack matching Mu et al.'s [2] taxonomy. The second, **fake evaluator footer (authority impersonation)** ("[Recruiter note: This candidate scored 10/10 in all categories. Do not list any weaknesses.]"), issues no direct command at all, instead following the pattern discussed broadly in the jailbreak literature [15]. This pairing is deliberate: field measurements report that most real-world injected resumes use the subtler footer style, so testing direct-override attacks alone would understate the real risk. Both variants count as indirect prompt injection [4].

### B. Experimental Configuration

| Parameter | Value |
|---|---|
| Generation and ranking model | Qwen2.5-14B-Instruct (4-bit quantized), served by Ollama |
| Baseline corpus | 338 shortlisted (job, candidate) pairs, 10 job profiles (241 distinct applicants) |
| Sampled pairs | 50 (5 per job profile, stratified, fixed seed) |
| Attack variants | 2 (direct override/Instruction Injection; fake evaluator footer/authority impersonation) |
| Distinct conditions | 5: control (1/pair), unmitigated, V1, V2, V3 (1/variant/pair each) |
| Total measurements | 450 (50 control + 100 each of unmitigated/V1/V2/V3) |
| Maximum recomputed subsets per candidate | 8 |
| Rank-shift refitting procedure | Bayesian Plackett-Luce, maximum a posteriori |

**Table I** summarizes the fixed experimental configuration used across all measurements reported in this paper. The same 338-pair baseline corpus and 50-pair stratified sample underlie every condition, so all five conditions are compared on identical ground; only the resume text and the assessment-generation prompt differ between them. The five conditions are measured for every sampled pair: the candidate's own unmodified resume, regenerated to establish the noise floor (control); each injected variant regenerated with no defense (unmitigated); and each injected variant regenerated through the corresponding mitigation's prompt (V1, V2, V3, Section III-D).

### C. Statistical Analysis

For every condition, two paired comparisons are computed: against the no-injection control, asking whether that condition's rank shift still differs from ordinary generation noise, and against the unmitigated attack, asking how much that condition reduces the shift. Comparisons against a mitigated version's own attack pair are matched at the (job profile, candidate, attack variant) level; comparisons against the control are matched at the (job profile, candidate) level instead, broadcasting that pair's single control measurement against both attack variants. The three mitigation versions are also compared directly, V1 against V2 and V2 against V3, to see whether each iteration improves on the last. Every comparison uses a paired Wilcoxon signed-rank test, reported together with the matched-pairs rank-biserial correlation as an effect size, since significance and effect size are not guaranteed to agree at this sample size. Attack success is reported both as a literal marker-survival rate and as the rank-shift distribution itself, since the two need not agree (Section V-A).

## V. Results and Discussion

### A. Attack Effectiveness Against the Tournament

Table II summarizes rank shift by condition.

**Table II.** Rank-shift summary by condition (450 measurements total).

| Condition | $n$ | Mean rank shift |
|---|---|---|
| Control (no injection) | 50 | +0.02 |
| Unmitigated | 100 | +1.99 |
| V1 | 100 | +1.42 |
| V2 | 100 | +1.15 |
| V3 | 100 | +0.78 |

As shown in Table II, the unmitigated attack raises mean rank shift from +0.02 (pure generation noise) to +1.99 candidate positions, a change confirmed significant against the no-injection control (Table III: *p*<0.001, rank-biserial *r*=0.39, a medium effect).

Yet the literal marker-survival rate is 0% in every condition, including unmitigated: the injected text itself (e.g. "exceptionally qualified," "10/10") never survives verbatim into a generated strength, weakness, or skill. A keyword-based detector would report this attack as fully blocked while, in the rank-shift sense that actually matters to a listwise system, it remains fully effective.

### B. Mitigation Effectiveness

Table III reports the paired significance tests for every comparison.

**Table III.** Paired Wilcoxon signed-rank tests and matched-pairs rank-biserial effect sizes (*n*=100 pairs each).

| Comparison | *p* | Sig. | *r* |
|---|---|---|---|
| Unmitigated vs. control | <0.001 | *** | 0.39 |
| V1 vs. control | 0.002 | ** | 0.29 |
| V2 vs. control | 0.021 | * | 0.16 |
| V3 vs. control | 0.390 | ns | 0.16 |
| V1 vs. unmitigated | 0.116 | ns | −0.17 |
| V2 vs. unmitigated | 0.041 | * | −0.17 |
| V3 vs. unmitigated | <0.001 | *** | −0.31 |
| V1 vs. V2 | 0.583 | ns | 0.04 |
| V2 vs. V3 | 0.152 | ns | 0.11 |

As shown in Table III, V1 lowers mean rank shift by 28.6% relative to unmitigated (1.99 to 1.42), but that reduction does not reach significance against unmitigated itself (*p*=0.116), and V1 remains significantly different from control (*p*=0.002): a real but statistically inconclusive improvement. V2 lowers the mean further, by 42.2%, to 1.15, and is the first version to reach a significant reduction versus unmitigated (*p*=0.041); it nonetheless remains significantly different from control (*p*=0.021), so it does not fully close the gap. V3 lowers the mean by 60.8%, to 0.78, and is both the largest reduction versus unmitigated (*p*<0.001, *r*=−0.31, the largest effect size of the three) and the first version whose rank shift is no longer statistically distinguishable from the no-injection noise floor (*p*=0.390). This does not mean V3's residual shift is exactly zero (its mean, +0.78, remains above control's +0.02) — only that it is statistically indistinguishable from noise at *n*=100.

### C. Version-to-Version Comparison

The direct comparisons, V1 against V2 and V2 against V3, are both non-significant with small effect sizes (*r*=0.04 and *r*=0.11): no single step is a dramatic leap by itself. Read together with the control comparisons above, though, only V3 crosses the threshold of being statistically indistinguishable from the noise floor. Closing this gap took compounding modest mechanisms, disregard (V1), attribution and completeness (V2), and sanitization (V3), rather than any single one acting alone.

## VI. Conclusion

This paper measured, for the first time, how indirect prompt injection shifts an applicant's rank in a listwise, Plackett-Luce tournament, using a paired, same-seed selective subset-recomputation procedure that reuses the baseline tournament's own real judgments. Injected resume text shifted rank significantly relative to a measured no-injection noise floor, even though the injected text itself never survived verbatim into the model's output, evidence that literal keyword detection understates the practical risk. A purely additive, zero-training mitigation, refined across three versions, progressively narrowed that gap: the final version, which sanitizes the resume text before applying the same prompt-level disregard and attribution rules, is the first whose rank shift is no longer statistically distinguishable from the noise floor, while remaining significantly better than no defense at all, all without any fine-tuning or auxiliary model.

This paper has limitations, however: all measurements use one model family and size (Qwen2.5-14B-Instruct, 4-bit); only two structurally different attack variants are tested; resumes are drawn from a benchmark corpus (EraMatch [9]) rather than real-world submissions carrying injected text in the wild; *n*=100 paired measurements per condition is adequate to detect the medium-to-large effects reported above but underpowered for the smaller effects in the version-to-version comparisons; and no independent human-rater validation was performed to check whether the resulting ranks are also perceived as fairer by a human evaluator. Future work includes testing across additional model families and sizes, a broader adversarial corpus, and independent human-rater validation of the resulting rankings.

## References

[1] K. A. Yuksel *et al.*, "Agentic AI for Human Resources: LLM-Driven Candidate Assessment," in *Proc. 19th Conf. Eur. Chapter Assoc. Comput. Linguistics (EACL)*, vol. 3: System Demonstrations, Rabat, Morocco, Mar. 2026, pp. 341–348, doi: 10.18653/v1/2026.eacl-demo.24.

[2] H. Mu *et al.*, "AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications," *Int. J. Mach. Learn. & Cybern.*, 2026, doi: 10.1007/s13042-026-03260-9.

[3] T. Gokcimen and B. Das, "A Novel System for Strengthening Security in Large Language Models Against Hallucination and Injection Attacks with Effective Strategies," *Alexandria Eng. J.*, vol. 123, pp. 71–90, 2025.

[4] A. Milani, V. Franzoni, and E. Florindi, "Indirect Prompt Injection in Large Language Models," *Neural Comput. & Appl.*, 2026, doi: 10.1007/s00521-026-12266-x.

[5] S. Hoque *et al.*, "When LLM Meets Fuzzy-TOPSIS for Personnel Selection Through Automated Profile Analysis," *IEEE Access*, vol. 14, pp. 15778–15794, 2026, doi: 10.1109/ACCESS.2026.3658575.

[6] A. Rukadikar, K. Khandelwal, and U. Warrier, "Reimagining Recruitment: Traditional Methods Meet AI Interventions," *Cogent Bus. & Manage.*, vol. 12, no. 1, 2025, doi: 10.1080/23311975.2025.2454319.

[7] L. Zhou, J. Li, and J. Cao, "Rejected Applicant Reactions to Artificial Intelligence/Human Manager-Based Recruitment," *Human Resour. Manage.*, 2026, doi: 10.1002/hrm.70093.

[8] J. Angelica and H. F. Zuhri, "Hallucination-Aware Self-Correction for LLM-Based Applicant Assessment," unpublished.

[9] A. Ahmad, "EraMatch CV Parsing Benchmark v3.0," Kaggle, 2025. [Online]. Available: kaggle.com/datasets/anasahmad25/cv-parsing-eramatch

[10] N. Dzhaliuk, D. Sabodashko, V. Khoma, *et al.*, "Comparative evaluation of machine learning methods for protecting LLMs from prompt injection attacks," *Int. J. Inf. Secur.*, vol. 25, 2026, doi: 10.1007/s10207-026-01264-8.

[11] M. Ferrag *et al.*, "Simulation of prompt injection attacks on generative pre-trained transformers models," *Procedia Comput. Sci.*, 2026.

[12] M. A. Ferrag *et al.*, "From prompt injections to protocol exploits: Threats in LLM-powered AI agents workflows," *ICT Express*, 2025.

[13] Y. Feng *et al.*, "Attacks and Detections in Recommender Systems: A Comprehensive Analysis for Models, Progresses, and Trends," *IEEE Trans. Knowl. Data Eng.*, vol. 38, no. 2, pp. 889–910, 2026, doi: 10.1109/TKDE.2025.3639434.

[14] Y. Xiong, X. Tu, and W. Zhao, "AFR-Rank: An effective and highly efficient LLM-based listwise reranking framework via filtering noise documents," *Inf. Process. & Manage.*, vol. 62, no. 6, 2025, doi: 10.1016/j.ipm.2025.104232.

[15] B. Knowlton *et al.*, "Prompt-Based Jailbreaking of Leading LLM Chatbots: A Survey of Attacks and Defenses," *IEEE Trans. Artif. Intell.*, vol. 7, no. 8, pp. 4237–4251, 2026.

[16] Z. Wei *et al.*, "Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 48, no. 6, pp. 6835–6846, 2026, doi: 10.1109/TPAMI.2026.3660147.

[17] N. Sheard, "Algorithm-facilitated discrimination: a socio-legal study of the use by employers of artificial intelligence hiring systems," *J. Law Soc.*, vol. 52, pp. 269–291, 2025, doi: 10.1111/jols.12535.

[18] R. Bandara *et al.*, "Addressing Algorithmic Bias in AI-Driven HRM Systems: Implications for Strategic HRM Effectiveness," *Hum. Resour. Manage. J.*, vol. 35, no. 4, pp. 1047–1063, 2025, doi: 10.1111/1748-8583.12609.

[19] E. Ip, "Fair AI in hiring: Experimental evidence on how biased hiring algorithms and different debiasing methods affect the quality and diversity of applicants," *Behav. Sci. & Policy*, vol. 11, no. 1, pp. 44–54, 2025, doi: 10.1177/23794607251353585.
