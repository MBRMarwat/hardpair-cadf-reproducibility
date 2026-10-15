# Submission-readiness audit

Target checked: Pattern Analysis and Applications, 2026-10-15.

## Outcome

The scientific audit is complete, the data/code evidence is locally reproducible, and the unarchived repository is public. The submission is **conditionally ready**, not yet administratively ready. No further model tuning is justified. The remaining blockers require author information, a selected code license, and permanent DOI archival rather than more experiments.

## Journal-specific requirements checked

- Scope fit: medical image pattern recognition is within scope.
- Analytical and experimental testing: supplied by the mathematical fusion checks, correction bounds, paired analyses, external cohort, and new same-condition comparator.
- Same-condition comparison: the new EfficientNet-B0 comparator uses the frozen HAM10000 split and preprocessing. It is a contemporary architecture comparator, not a reproduction of every published method.
- Replicable research: executable code, configs, seeds, split IDs, checkpoints, source hashes, prediction tables, and evaluation scripts exist.
- Public repository: **satisfied** at <https://github.com/MBRMarwat/hardpair-cadf-reproducibility>. It contains unarchived code/data evidence and an explanatory README. Permanent DOI archival of release `v1.0-submission` remains pending.
- Editable source: the current manuscript is a DOCX, but it contains only Introduction, Literature Review, and References. Methods, Results, Discussion, Conclusion, abstract, title page, and declarations still need to be assembled in the manuscript.
- Abstract: must be 150-250 words.
- Keywords: provide 4-6.
- References: numbered citations and full DOI links.
- Data Availability Statement: mandatory.
- LLM disclosure: generative assistance should be documented in Methods unless the authors determine that use was limited to copy editing.
- Figures: captions, accessibility, dimensions/resolution, and sequential citation must be checked in the final manuscript.

## Author-supplied items still required

1. author names, affiliations, corresponding email, and ORCIDs;
2. author-contribution statement;
3. funding statement and grant numbers, or an explicit no-funding statement;
4. competing-interests statement;
5. ethics/consent wording appropriate to secondary analysis of public de-identified datasets and the authors' institution;
6. selected code license;
7. permanent DOI for release `v1.0-submission` (the public, unarchived repository URL is now available);
8. confirmation of the ranking system required by the author's institution (SCImago/SJR and Clarivate JCR quartiles are not interchangeable).

## Draft statements requiring author review

### Data availability

HAM10000 is publicly available through Harvard Dataverse at <https://doi.org/10.7910/DVN/DBW86T> and is described at <https://doi.org/10.1038/sdata.2018.161>. BCN20000 is publicly available through Figshare at <https://doi.org/10.6084/m9.figshare.24140028> under CC BY 4.0 and is described at <https://doi.org/10.1038/s41597-024-03387-w>. The exact lesion-disjoint split identifiers, derived predictions, summary data, and analysis scripts supporting this study are available at <https://github.com/MBRMarwat/hardpair-cadf-reproducibility>. The permanent archive DOI for release `v1.0-submission` will be added before publication. Original dermoscopic images are not redistributed in the study repository.

### Code availability

Executable training, checkpoint-resume, inference, calibration, fusion, bootstrap, audit, and figure-generation code, together with frozen configurations and environment information, are available at <https://github.com/MBRMarwat/hardpair-cadf-reproducibility>. The release tag used for the manuscript is `v1.0-submission`.

### Ethics

This study is a secondary analysis of publicly available, de-identified dermoscopic image datasets. No new participants were recruited and no new identifiable data were collected. **[THE AUTHORS/INSTITUTION MUST CONFIRM WHETHER FORMAL ETHICS EXEMPTION OR REVIEW WAS REQUIRED AND INSERT THE APPROVED WORDING.]**

### Competing interests

The authors declare **[NO COMPETING INTERESTS / INSERT FULL DISCLOSURE]**.

### Funding

This work was supported by **[FUNDER AND GRANT]** / **[The authors received no specific funding for this work]**.

### AI-assistance disclosure

Generative AI tools were used to assist with code generation, debugging, analysis scripting, and preparation of draft text. All experimental decisions, source verification, execution, numerical results, and final manuscript content were reviewed and remain the responsibility of the human authors. **[EDIT THIS STATEMENT TO MATCH THE AUTHORS' ACTUAL USE AND JOURNAL POLICY.]**

## Stopping rule

Do not add more gates, alpha/beta sweeps, architectures, or test-guided tuning. Further experiments should be prompted only by a discovered validity problem or specific reviewer request. Publishing the repository and completing the manuscript/declarations are now higher priority than additional modeling.

