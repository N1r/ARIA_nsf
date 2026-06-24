# ARIA-NSF Revision Roadmap

Revision mode: pre-submission revision using `academic-paper-reviewer` + `academic-paper` style workflow.  
Date: 2026-06-12  
Manuscript: `aria_nsf_conference_paper.md`

## Simulated Editorial Decision

**Decision: Major Revision before submission.**

The manuscript has a promising interdisciplinary contribution, but the first draft reads partly as a concept note. For a speech-science / speech-technology conference, the paper needs a clearer claims hierarchy: what is already implemented, what is pilot observation, what remains planned evaluation, and why this is still worth presenting.

## Reviewer Configuration

| Reviewer | Perspective |
|---|---|
| EIC | Fit for speech science / speech technology venue |
| R1 Methodology | Evidence strength, evaluation, baseline design |
| R2 Domain | Phonetics, formant synthesis, experimental stimulus validity |
| R3 ML/DSP | Model specification, source-filter logic, reproducibility |
| Devil's Advocate | Overclaiming, missing alternatives, “so what?” test |

## Required Revisions

| # | Issue | Severity | Resolution |
|---|---|---|---|
| R1 | Abstract overstates pilot findings without enough metrics. | Major | Rewrite abstract to frame the work as a system/pilot paper and separate implemented design from future validation. |
| R2 | Paper lacks explicit design requirements for phonetic stimulus generation. | Major | Add a Design Requirements section bridging phonetics and ML readers. |
| R3 | Evaluation is too late and too hypothetical. | Major | Add concrete baselines, ablations, and acoustic/perceptual evaluation matrix. |
| R4 | Pilot section needs to read like a reproducible experiment plan, not informal progress notes. | Major | Rename and restructure as Pilot Corpus and Experimental Conditions. |
| R5 | Learned biquads create a potential controllability risk that needs sharper treatment. | Major | Discuss parameter coupling and add ablation requirements. |
| R6 | Contribution statement should clarify that this is not competing with large-scale HiFi-Glot on generality. | Minor | Reframe contribution as small-data, single-speaker, high-control stimulus generation. |

## Revision Actions Completed

| # | Action | Status |
|---|---|---|
| 1 | Created revised manuscript file rather than overwriting original. | RESOLVED |
| 2 | Revised title, abstract, and contribution framing. | RESOLVED |
| 3 | Added Design Requirements section. | RESOLVED |
| 4 | Reworked pilot section into corpus + experimental conditions. | RESOLVED |
| 5 | Added baseline and ablation table. | RESOLVED |
| 6 | Strengthened evaluation protocol and limitations. | RESOLVED |
| 7 | Regenerated PDF and LaTeX preview. | RESOLVED |

