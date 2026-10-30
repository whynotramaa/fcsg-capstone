# Short project-report template

This is an outline and LaTeX starter, not a written report. Start with `main.tex`. The supplied NITR class, logo, and IEEE bibliography style are copied from the college package. The local class has one title-page fix to remove tabular padding that exceeded the text width. A PDF copy of the EPS logo is included for compilation. The original download is unchanged.

## Proposed short structure

1. **Project overview**: problem, objective, what we are doing, scope.
2. **Proposed pipeline**: one architecture diagram, component table, and brief training objective.
3. **Evaluation plan and current status**: data, baselines, metrics, completed/pending work, and next steps.
4. **References**: verified papers actually cited, generated with BibTeX. This is unnumbered.

Aim for roughly 5–8 pages of main content and references, excluding institution-required front matter. This is an editorial suggestion, not a college page limit. A proposal/overview can use this short structure. Confirm the expected depth before using it as a final completed-project report.

Add a separate results chapter only if the submission requires experimental findings.

## Supplied college requirements

Source: `Chap01/Guidelines.tex` and `FrontPages.tex` in the supplied package, whose class identifies itself as v4.5 dated 2016-10-07. These supplied requirements were not checked against current institutional policy.

- A4; left margin 31 mm; other margins 25 mm. Preserve the class layout.
- Times New Roman 12 pt or Arial 11 pt. One font family throughout, including diagram labels.
- Body spacing 1.5. Contents, tables, references, and other listed exceptions use single spacing.
- Table captions above; figure captions below. Chapter-based numbering.
- Abstract at most two pages, followed by 3–7 keywords.
- Cover excluded from page count; preliminary pages counted in Roman numerals, with displayed numbers beginning at contents; chapters start at Arabic 1.
- Required front pages in the supplied guidelines include cover, title, examination certificate, supervisor certificate, originality declaration, abstract, and contents. Dedication and acknowledgment are optional. Keep certificates for a final report; confirm their applicability to a short progress submission.
- Use one references section. The guidelines request all author names in reference entries. Inspect the bibliography output because a selected journal style can abbreviate authors.

The starter assumes B.Tech, CSE, and one supervisor based on repository context. Names, roll number, date, department/degree details, supervisor count, and certificate applicability need confirmation. The sample class is oriented toward a single author; a team submission may require an approved front-page adaptation. Never retain the sample identities or invent signatories.

The supplied instructions conflict on appendix ordering: one enumerated list places appendices before references, while prose requests references before appendices. This short template omits appendices.

## Sources for the writer

| Section | Repository evidence |
|---|---|
| Scope and project history | `../README.md`, `../results/milestone_status.md` |
| Implemented pipeline | `../models/fcsg.py`, `../models/blocks.py` |
| Degradation order and ranges | `../src/fcsg_net/degrade.py` |
| Training and losses | `../training/train.py`, `../configs/` |
| Completed benchmark | `../results/phase3/*/evaluation.json` |
| Timing and routing diagnostics | `../results/diagnostics/summary.json`, `../results/diagnostics/routing.json` |
| Literature starting points | `../notes/related.md` |

Treat `../plan.md` as historical intent. Treat the technical report in `../output/pdf/` as a derived summary, not the authority when raw evidence differs. Phase 4 jobs and the report's live snapshot are historical progress records, not verified current job states or final ablation outcomes.

## Pipeline drawing brief

Show the data preparation path separately from inference:

- Training pairs: clean DIV2K RGB image → Gaussian blur → downsample and upsample → Gaussian noise and clipping → JPEG → degraded input, paired with clean target.
- Restoration: degraded RGB → complementary low/mid/high FFT bands → three shared spatial experts → per-band spatial top-2 weighted mixing → SE fusion → refinement → input residual addition → restored RGB.
- Training only: output and clean target feed Charbonnier and frequency losses; dense routing probabilities feed the decaying entropy term.

The experts have 7×7, 5×5, and 3×3 kernels. They are shared across bands, not nine independently parameterized experts. All experts execute. The method uses sparse mixing weights, not conditional computation. Show the gate feeding the mixture weights and the input skip reaching the final addition.

## Claim limits

Completed Phase 3 evidence reports 97,088 FCSG-Net parameters, 9.39 reported GFLOPs, and 27.24 dB on the 100-image composite validation protocol. FFDNet has better restoration scores and lower measured latency. Do not claim state-of-the-art accuracy or speed.

Routing maps and correlations support an exploratory analysis. Inference substitutions do not replace a trained ablation. The local files do not establish completed results for the fresh 100k-step reference/uniform experiment. Avoid live progress curves in the short report.

## Keep and remove

Keep one pipeline diagram, a short objective, a component table, a protocol summary, and relevant citations. If results are required, include one compact final-evidence table and an honest interpretation.

Remove the college's sample guideline chapters, sample references, fictional identities, dissemination page, index, duplicate bibliography, full source listings, long training logs, all 100 per-image results, unrelated literature surveys, and unverified novelty claims. Do not remove required institutional front matter merely to shorten the document.

## Writing handoff

For later actual writing, the requested first choice is **MiMo v2.6 Flash Free through OpenCode**, with **Claude Haiku** as fallback. Exact provider model identifiers and availability must be checked before delegation. No drafting model has been called for this template.

Write from the students' technical notes and verified evidence. Cite borrowed ideas and paraphrased claims. Use quotation marks for exact quotations. Keep dataset, protocol, and result provenance. Check every citation against its primary source before adding its BibTeX entry. Do not fill references with uncited papers just to increase their count.

No AI-detector or similarity percentage is guaranteed. Accurate attribution and original, project-specific explanations are the review criteria. Follow the institution's applicable disclosure rules.

## Build and check

From this directory, after adding cited BibTeX entries and replacing the references placeholder block with the bibliography command:

```sh
xelatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

The system XeLaTeX installation is incomplete. The template was compiled successfully with Tectonic 0.17.0 using genuine Times New Roman fonts installed for the current user. All pages were rendered and visually inspected. The final PDF is `../output/pdf/fcsg_report_template.pdf`. A full XeLaTeX installation with Times New Roman can also build the source. With Tectonic installed, run `tectonic main.tex` from this directory. Do not silently substitute the institutional font.

Before submission, inspect the rendered PDF for placeholder text, certificate wording, numbering, font consistency, clipping, references, and figure legibility. The empty bibliography is intentional until verified cited sources are supplied.

The PDF intentionally includes visible authoring placeholders. The references page must be replaced with the generated bibliography when writing begins. Confirm author pronouns in `FrontPages.tex`; the supplied certificate currently uses the sample male setting. No model was delegated to write report prose.
