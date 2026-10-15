# Course-provided donor scenario data

The two files were supplied inside `midterm.zip` alongside an unfilled notebook template. The completed notebook was supplied separately. The portfolio bundles losslessly compressed CSVs and verifies their uncompressed SHA-256 checksums before model fitting.

| File | Rows | Content |
| --- | ---: | --- |
| `train.csv.gz` | 27,132 | Demographics, income label, prior-contact indicator, and partially observed past-donation labels |
| `test.csv.gz` | 18,090 | Corresponding features plus future-donation labels and scenario latitude/longitude |

The exercise describes a fictional nonprofit and geographic outreach scenario. Demographic source provenance and a separate dataset license were not included in the supplied ZIP; no additional license is assigned here. The portfolio preserves the supplied data rather than claiming a real nonprofit deployment. See `manifest.json` for exact checksums and byte lengths.

The curated experiment reads these compressed files directly; do not manually unpack or edit them for the default run. The original notebook source expects uncompressed files named `train.csv` and `test.csv` in its working directory.
