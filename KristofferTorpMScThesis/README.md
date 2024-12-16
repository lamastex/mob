# Kristoffer Torp's Mastes Thesis 

## in Mathematics 2024 Uppsala University.

```
$ tree .
.
├── abstract.tex
├── chapters
│   ├── discussion.tex
│   ├── results.tex
│   ├── the_generalized_jaccard_index.tex
│   └── trajectories_and_co-trajectories.tex
├── graphics
│   ├── 1024.png
│   ├── 128.png
│   ├── 256.png
│   ├── 512.png
│   ├── 64.png
│   ├── Full.png
│   ├── differences.png
│   ├── non-results
│   │   ├── DTW.png
│   │   ├── euclidean lock-step.png
│   │   ├── euclidean_dtw.png
│   │   ├── hausdorff.png
│   │   ├── non-weighted.png
│   │   └── trajectory.png
│   └── results
│       ├── Jaccard_Thu.png
│       ├── L1_Jaccard_Feb.png
│       ├── L1_Thu.png
│       ├── antenna_distribution.png
│       ├── comparison_minhash.png
│       ├── comparison_minhash_small.png
│       ├── movement profiles.png
│       ├── new_L1_Jaccard_Feb.png
│       ├── new_L1_Jaccard_Year.png
│       └── population_movement.png
├── introduction.tex
├── main.aux
├── main.bbl
├── main.blg
├── main.log
├── main.pdf
├── main.tex
├── main.toc
└── thesis.bib

4 directories, 37 files
```

# To make PDF Do

- pdflatex main.tex
- bibtex main
- pdflatex main.tex
- pdflatex main.tex

Resulting PDF in main.pdf
