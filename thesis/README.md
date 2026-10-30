# FCSG-Net final report

This folder holds the B.Tech project report, built on the NIT Rourkela
`DissertationGuidelinesV4_5` template. All chapters, appendices, the abstract
and the acknowledgement are written. The repository is the source for every
number; `\evidence{}` markers in the .tex point at the file behind a table.

Scope: Phases 1 to 3 (verification, FFDNet reproduction, matched composite
training of FCSG-Net, DnCNN and two FFDNet variants) plus the routing
diagnostics run on the Phase 3 checkpoints. The Phase 4 routing ablation is
left out and appears only as future work.

Authors (ascending roll number): Akshat Mittal (123CS0954), Risav Rathi
(123CS0957), Ramnath Thakur (123CS0965). Supervisor: Dr. Shyampada Mukherjee.

Build:

```
cd thesis && latexmk -pdf -outdir=build main.tex
```

`NITR.cls` and `FrontPages.tex` here are stand-ins written because the
college bundle did not include them. Replace both with the department's
versions when available and copy our field values across.

---

The rest of this file is the original planning note, kept for reference.

# FCSG-Net final report: plan and scaffold

This folder is the final B.Tech report, laid out on the NIT Rourkela
`DissertationGuidelinesV4_5` template. Structure, headings, figure slots and
the hard numbers are in place. The prose is not: every gap is marked
`\todo{...}` and prints in red, and every chapter opens with a comment block
telling the writer what goes in, how long it should be, and which repo files
hold the evidence.

## 1. Before anything compiles

The college gave us only the main `.tex` and the logo. The template loads
files that were not in what we received:

| Missing file | Why it is needed | Action |
|---|---|---|
| `NITR.cls` | `\documentclass{NITR}`, defines `\coverPage`, `\titlePage`, certificates | Get the full template zip from the department / Overleaf |
| `FrontPages.tex` | Holds title, names, roll numbers, guide(s), dept, date, abstract, acknowledgement text | Copy from the zip, fill our details |
| `Ref/IEEEtran.bst` | Reference style | Comes with the zip, or any TeX Live install |
| `fancyhdr`, `setspace` etc. | Probably loaded inside `NITR.cls` | Nothing, if the class is in place |

`NITlogo.eps` is already here. With pdfLaTeX it needs `epstopdf` (on by
default in TeX Live and Overleaf). Easiest path: upload this whole folder plus
the college zip to Overleaf. Note that `main.tex` reads figures from
`../output/pdf/fcsg-report-assets/`, so on Overleaf either upload the repo root
or copy the PDFs you use into `thesis/figures/`.

## 2. Folder layout

```
thesis/
├── main.tex                    driver, follows college template order
├── macros.tex                  \todo, \evidence, \fcsg, \expert{k}
├── NITlogo.eps
├── FrontPages.tex              <-- from college zip (missing)
├── NITR.cls                    <-- from college zip (missing)
├── chapters/
│   ├── abbreviations.tex
│   ├── ch1_introduction.tex
│   ├── ch2_literature.tex
│   ├── ch3_method.tex
│   ├── ch4_setup.tex
│   ├── ch5_results.tex
│   ├── ch6_routing.tex
│   └── ch7_conclusion.tex
├── appendix/
│   ├── appA_reproducibility.tex
│   └── appB_per_image.tex
├── Ref/
│   ├── references.bib          15 entries, keys already cited
│   └── IEEEtran.bst            <-- from college zip (missing)
├── figures/                    only for NEW figures; existing ones are
│                               read from ../output/pdf/fcsg-report-assets/
└── college_reference/
    └── DissertationGuidelinesV4_5.tex   untouched original, for diffing
```

## 3. Document flow

```
 FRONT MATTER (roman pages, mostly college boilerplate)
 ┌────────────────────────────────────────────────────────────┐
 │ Cover → Title → Cert. of Examination → Supervisor Cert.     │
 │ → Dedication → Declaration → Acknowledgement → Abstract     │
 │ → Contents → List of Figures → List of Tables → Abbrev.     │
 └────────────────────────────────────────────────────────────┘
                              │
 MAIN MATTER (~50-60 pages)   ▼
 ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
 │ Ch1 Intro│→ │ Ch2 Lit. │→ │ Ch3      │→ │ Ch4 Setup│
 │ question │  │ position │  │ Method   │  │ + checks │
 │  5-7 pp  │  │  7-10 pp │  │ 10-12 pp │  │  6-8 pp  │
 └──────────┘  └──────────┘  └──────────┘  └────┬─────┘
                                                │
        ┌───────────────────────────────────────┘
        ▼
 ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
 │ Ch5 Results  │→ │ Ch6 Routing  │→ │ Ch7 Concl.   │
 │ quality/cost │  │ + ablation   │  │ limits/future│
 │   8-10 pp    │  │   7-9 pp     │  │   3-4 pp     │
 └──────────────┘  └──────────────┘  └──────────────┘
                              │
 BACK MATTER                  ▼
   References → App A (reproducibility, hashes) → App B (per-image PSNR)
```

## 4. The story the report has to tell

FCSG-Net loses on PSNR by about 1 dB to every baseline and is also the
slowest on the T4 despite having the fewest FLOPs. That is the outcome. The
report is credible only if it is built around that, not around hiding it.

```
          what we set out to test                 what we can actually show
  ┌──────────────────────────────────┐   ┌──────────────────────────────────────┐
  │ can a <5M, <10 GFLOP model split │   │ yes, 97k params, 9.39 GFLOPs, all     │
  │ bands and route per pixel?       │──▶│ verification gates passed            │
  ├──────────────────────────────────┤   ├──────────────────────────────────────┤
  │ is it competitive?               │──▶│ -0.59 dB vs DnCNN with 5.7x fewer    │
  │                                  │   │ params; behind FFDNet; slower in ms  │
  ├──────────────────────────────────┤   ├──────────────────────────────────────┤
  │ does the router learn something? │──▶│ maps vary spatially; r=+0.81 with     │
  │                                  │   │ noise sigma; overriding costs 0.37 dB │
  ├──────────────────────────────────┤   ├──────────────────────────────────────┤
  │ does learning a router help?     │──▶│ OPEN: depends on the 100k ablation    │
  └──────────────────────────────────┘   └──────────────────────────────────────┘
```

So the headline is a trade-off and an interpretability study, not a new SOTA.
The single most important figure is quality against parameters (and against
latency), not a bare PSNR bar chart.

## 5. What to include and what to leave out

The technical report (`output/pdf/fcsg_technical_report_2026-10-01.md`) is 29
pages written for us, not for an examiner. It is the raw material. Mapping:

| Material | Decision | Where |
|---|---|---|
| Problem, degradation model, architecture, loss | Include in full | Ch1, Ch3 |
| Design reasoning from `plan.md` 0.2 to 0.4 (no complex spectra to CNNs, normalised radii, spatial not global gate) | Include, it is the best "why" material we have | Ch3 |
| `notes/related.md` | Include, adapt to LaTeX with citations | Ch2 |
| Verification gates (FFDNet repro, band-sum, overfit, resume) | One table, one paragraph | Ch4 |
| Phase 1 Gaussian DnCNN result, clipped vs unclipped noise detail | One sentence as a sanity check | Ch4 |
| Main composite comparison table | Include, numbers prefilled | Ch5 |
| Latency paradox (fewest FLOPs, slowest) | Include as a finding with a reason | Ch5 |
| Tiling sensitivity, memory | Short subsection | Ch5 |
| Routing maps, correlations, interventions | Include, this is the contribution | Ch6 |
| Matched routing ablation | Include if finished; else interim + future work | Ch6 |
| Limitations table | Rewrite as prose | Ch7 |
| SHA-256 hashes, source bundles, environment | Appendix only, 16-char prefixes | App A |
| Per-image PSNR (100 rows x 4) | Appendix, generated by script; drop if page-limited | App B |
| Kaggle mechanics, quota, session limits | One or two sentences in training protocol | Ch4 |
| Diagnostic notebook v1/v2 crashes, Kaggle debugging history | Leave out | none |
| "Implementation issues and corrections" table | Leave out, keep only the two that change interpretation (crop vs full-image PSNR, sparse weights vs sparse compute) | Ch4, Ch2 |
| Interim live ablation curve (6k steps) | Leave out once final numbers exist | none |
| Planned ablations A1, A3, A4, A5 that never ran | Future work only, never as results | Ch7 |
| Template sample chapters (`Guidelines`, `Unstructured`), ASME footnote, sample dissemination | Removed | none |
| Index page | Removed unless someone tags terms with `\index{}` | none |

## 6. Rules for the writer

1. **Numbers come from files, not memory.** Every figure in the text must
   trace to `results/` or the technical report. Put the source in
   `\evidence{path}` next to the table so a reviewer can check it. The tables
   already filled in were copied from `results/benchmark.csv`,
   `results/milestone_status.md` and `results/diagnostics/`.
2. **Never mix protocols.** 16-crop training validation (30.59 dB) and
   100-image full evaluation (27.24 dB) are different measurements. Published
   paper numbers never go in our tables.
3. **Banned claims:** "novel", "first", "state of the art", "outperforms",
   "experts specialise in texture/edges", "sparse computation". The code
   computes all experts; the gate gives sparse weights only.
4. **Correlation is not causation.** Chapter 6 numbers come from 10 source
   images, 36 exploratory correlations, no correction. Say this beside them.
5. **Interventions are not ablations.** Overriding weights at inference shows
   the checkpoint depends on its gate. Only the trained pair shows whether
   having a gate helps.
6. **Figures:** vector PDFs from `output/pdf/fcsg-report-assets/`. Every
   figure is referenced in the text before it appears and has a caption that
   reads on its own.
7. **Done means:** `grep -rn '\\todo' thesis/` returns nothing and the PDF
   has no red text and no `??` references.

## 7. Open decisions (need a human)

| Decision | Options | Notes |
|---|---|---|
| Supervisor certificate | `\certificateOneSupervisor` or `\certificateTwoSupervisors` | `main.tex` line with `DECIDE` |
| Routing ablation status | Final numbers, or report interim | `results/phase4/jobs.json` still says RUNNING as of 1 Oct; pull the Kaggle outputs first |
| Reference style | IEEEtran (set) or another from the template | IEEEtran is standard for CS/EE |
| Appendix B | Keep or drop | Depends on any page limit the department sets |
| Dissemination page | Keep only if a paper/poster exists | Commented out |
| Report title | Proposal title vs one that reflects the trade-off framing | Lives in `FrontPages.tex` |

## 8. Small engineering left before writing

- Pull and evaluate `fcsg_reference` and `fcsg_uniform` (100 images, PSNR,
  SSIM, LPIPS) and fill `tab:ablation` in Ch6.
- Check `benchmark.pdf` actually plots PSNR against params. If not, add a
  short script in `evaluation/` that draws PSNR vs params and PSNR vs latency
  from `benchmark.csv` and `results/diagnostics/summary.json`.
- Script for Appendix B: read `results/phase3/per_image_scores.csv`, write
  LaTeX rows into `appendix/per_image_rows.tex`, wrap in `longtable`.
- Build the qualitative 2x2 panel from the four `qualitative_*.pdf` files.

## 9. Suggested order of writing

```
 week of 2 Nov    Ch3 Method, Ch4 Setup    (facts we already know)
                  pull ablation results, run missing scripts
 week of 9 Nov    Ch5 Results, Ch6 Routing (once ablation numbers are in)
                  Ch2 Lit (mostly adapting notes/related.md)
 by ~13 Nov       Ch1 Intro, Ch7 Conclusion, Abstract (written last,
                  because they summarise what the others say)
 last 2-3 days    front pages, proofreading, \todo sweep, PDF check
```

Writing the intro and abstract last is deliberate: they have to promise
exactly what Chapters 5 and 6 deliver, and nobody knows that until those are
written.
