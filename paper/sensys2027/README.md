# SenSys 2027 draft (Overleaf)

Written for: the authors. Plan: `../../research/SENSYS_2027_PLAN.md`.

## Upload to Overleaf

1. In Overleaf, choose **New Project > Upload Project** and upload this folder as a zip, or upload the files individually keeping the `sections/` subfolder.
2. Main document: `main.tex`. Compiler: **pdfLaTeX**.
3. `acmart` is included in Overleaf's TeX Live. No extra packages beyond `acmart` are needed; `booktabs`, `xcolor` and `amsmath` are standard.

## Before submission

- [ ] Remove every `\TODO{...}`. The red placeholders must not appear in the submitted PDF.
- [ ] Switch to the review build only: `\documentclass[sigconf,review,anonymous]{acmart}` (already set). Do not add names, affiliations or emails.
- [ ] Anonymize artifact links: use an anonymous link (for example anonymous.4open.science). The original GitHub user name and cluster paths must not appear.
- [ ] Strip PDF metadata (author, title, keywords) before upload.
- [ ] Verify every bibliography entry: venue, year, pages, arXiv ID. The `references.bib` entries were written from memory and the uploaded PDFs; they have not all been checked against the publishers.
- [ ] Check the page limit: 12 pages of body text (references and appendices are not counted).
- [ ] Describe any generative AI used to conduct the research in the methods section.
- [ ] Check ACM author-rights terms for any figure or table reused from the SEC '25 paper.
- [ ] Refer to the prior paper in the third person only (CFP requirement).

## Layout

- `main.tex`: document class, front matter, section inclusion.
- `references.bib`: bibliography.
- `sections/abstract.tex`, `introduction.tex`, `related.tex`, `design.tex`, `evaluation.tex`, `discussion.tex`, `extension_vit.tex` (optional, delete if space is short), `conclusion.tex`: content (SLM scope).
- Figures: none yet. Add an `figures/` folder when the architecture and results figures exist. Do not copy figures from the SEC '25 paper without permission.
