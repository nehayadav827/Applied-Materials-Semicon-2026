# References

Public sources supporting the semiconductor structures, layout geometry,
and pattern-density choices used by `generate_dataset.py`. These sources
justify the *style* of the synthetic imagery -- pattern geometry, density,
periodicity -- not literal numeric image-generation parameters (pitch
values, noise sigma, etc.), which were chosen independently to produce
varied, controllable difficulty.

## Source list

| Key | Source |
|---|---|
| irds2017 | IRDS 2017 More Moore roadmap -- https://irds.ieee.org/images/files/pdf/2017/2017/IRDS%20MM.pdf |
| irds2024 | IRDS 2024 More Moore roadmap -- https://irds.ieee.org/images/files/pdf/2024/2024IRDS%20MM.pdf |
| itrs2015 | ITRS 2015 More Moore roadmap -- https://www.semiconductors.org/wp-content/uploads/2018/06/5_2015-ITRS-2.0_More-Moore.pdf |
| ibm_finfet14 | IBM Research, "Opportunities and Challenges of FinFET as a Device Structure Candidate for 14nm Node CMOS Technology" -- https://research.ibm.com/publications/opportunities-and-challenges-of-finfet-as-a-device-structure-candidate-for-14nm-node-cmos-technology |
| semieng_7nm | Semiconductor Engineering, "7nm Fab Challenges" -- https://semiengineering.com/7nm-fab-challenges/ |
| freepdk15 | FreePDK15 predictive PDK paper -- https://arxiv.org/pdf/2009.04600 |
| arxiv_ncfinfet | arXiv 2007.14448 (NC-FinFET / IRDS last-FinFET-node context) -- https://arxiv.org/pdf/2007.14448 |
| ti_wavy_bitline | TI patent EP0780901A2 (arcuate moats / wavy bitlines) -- https://patents.google.com/patent/EP0780901A2/en |
| hynix_dram | EE Times, "Hynix DRAM layout, process integration adapt to change" -- https://www.eetimes.com/hynix-dram-layout-process-integration-adapt-to-change/ |
| sram_6t | US Patent 5,554,874 (6T SRAM cell) -- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5554874 |
| std_cell | US Patent 6,938,226 (7-track standard cell library) -- https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6938226 |
| imec_logic | imec, "View on logic technology roadmap" -- https://www.imec-int.com/en/articles/view-logic-technology-roadmap |
| imec_damascene | imec, semi-damascene interconnect announcement -- https://www.imec-int.com/en/articles/imec-demonstrates-semi-damascene-interconnects-fully-self-aligned-vias-18nm-metal-pitch |
| ibm_beol | IBM Research BEOL blog (IEDM Cu interconnects) -- https://research.ibm.com/blog/beol-cu-interconnects-iedm |

## Which sources justify which generated pattern style

| Pattern style (`matching.py` / `generate_dataset.py`) | Citation keys | Justification |
|---|---|---|
| `legacy_dram_1x` | hynix_dram | Regular repeating memory-cell dot array, motivated by DRAM layout regularity. |
| `legacy_finfet` | ibm_finfet14, arxiv_ncfinfet | Parallel fin lines + crossing gate, motivated by FinFET device geometry. |
| `dram_staggered_realistic` (`realistic_dram_staggered`) | hynix_dram, sram_6t | Staggered repeated storage-node array, motivated by DRAM/SRAM repeated-cell geometry. |
| `finfet_via_realistic` | ibm_finfet14, arxiv_ncfinfet, irds2024 | Fin-line grid with contact/via bands, motivated by FinFET + advanced-node interconnect context. |
| `wavy_dram_bitline` | ti_wavy_bitline, hynix_dram | Non-perfectly-straight (arcuate) repeated bitline/dot geometry, motivated by patent literature on curved memory-array structures. |
| `mesh_grid` | imec_damascene, ibm_beol | Dense orthogonal line + via grid, motivated by interconnect mesh density context. |
| `ring_array` | irds2017, irds2024 | Dense repeated ring/contact geometry, motivated by scaling-driven pattern density. |
| `contact_array_square` / `contact_array_round` | imec_logic, irds2017 | Regular contact/via array, motivated by advanced-node contact density. |
| `layered_cell` | hynix_dram | Word-line/bit-line/storage-contact layered stack, motivated by DRAM cell process integration. |
| `logic_stripes` | imec_logic, std_cell | Alternating-width stripe stack, motivated by logic/standard-cell routing texture. |
| `beol_interconnect` | imec_damascene, ibm_beol | Multi-layer metal/via BEOL stack, motivated by semi-damascene and Cu interconnect process descriptions. |
| `standard_cell_regular` | std_cell, imec_logic | Fixed-pitch cell tracks bounded by row/cell-boundary lines, motivated by standard-cell library regularity. |

## Noise/degradation model justification

The SEM imaging degradation pipeline (`generate_dataset.py`: beam blur,
astigmatism, shot noise via Poisson statistics, charging streaks,
salt-and-pepper noise, speckle, raster drift/jitter, edge-brightening)
follows commonly-documented SEM image-formation effects discussed in the
hackathon's own problem-statement sessions (Applied Materials Drift-Sense,
Key Concepts & Q&A, 6 August 2026) and general SEM imaging literature.
Specific numeric parameters (blur sigma, noise amplitude, dose scale) were
chosen independently to produce a controllable difficulty range, not taken
directly from a cited source -- consistent with the guidance that these
sources justify pattern *style*, not literal image-generation numbers.

This file is generated from the citation registry embedded in
`generate_dataset.py` (`CITATIONS` and `STYLE_CITATION_MAP` dicts) so it
always matches what the code actually implements.
