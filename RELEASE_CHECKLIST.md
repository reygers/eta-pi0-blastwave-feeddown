# Publication and Archival Checklist

This checklist records the repository state associated with the accepted manuscript.

## Completed

- public-facing workflow and dependency list are documented in `README.md`
- accepted-paper code is tagged `journal-submission-v2.1-post-acceptance`
- GitHub citation metadata are provided in `CITATION.cff`
- the accepted-paper release contains the two baseline HDF5 kernels required by the documented workflow
- generated figures, intermediate outputs, local environments, and LaTeX build products remain ignored
- the repository uses the MIT License

## Before Announcing the Repository

- make the GitHub repository public
- verify the release and both kernel assets are visible without authentication
- update the paper citation in `README.md` and `CITATION.cff` when the journal DOI, volume, and article number are assigned

## Optional Follow-up

- archive the release with Zenodo and add the resulting software DOI to `CITATION.cff`
- run the complete workflow once from a clean environment using `requirements.txt`
- review third-party input-data attribution and redistribution terms
