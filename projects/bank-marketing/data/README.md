# Dataset attribution

This project uses the `bank-additional-full.csv` variant of **Bank Marketing**, distributed by the UCI Machine Learning Repository. UCI describes this variant as 41,188 examples with 20 input variables, ordered from May 2008 to November 2010. The extra `y` column is the subscription outcome.

- Dataset source: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Dataset DOI: https://doi.org/10.24432/C5K306
- Creators: S. Moro, P. Rita and P. Cortez.
- Dataset license: Creative Commons Attribution 4.0 International (CC BY 4.0), as stated on the UCI dataset page.
- License terms: https://creativecommons.org/licenses/by/4.0/

The supplied data is unchanged and only gzip-compressed for a smaller repository. No authorship or ownership of the dataset is claimed by this project. This data attribution does not grant a separate license to project code or the original report.

Uncompressed filename: `bank-additional-full.csv`  
Size: 5,834,924 bytes  
Delimiter: semicolon  
SHA-256: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`

The curated experiment reads the compressed file directly and checks the uncompressed hash against `manifest.json` before fitting. Run `python -m decision_analytics.bank` from the repository root after installing the project. The archived original notebook instead expects an uncompressed `bank-additional-full.csv` in its working directory.
