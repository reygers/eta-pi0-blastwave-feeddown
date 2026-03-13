# GitHub Release Checklist

This checklist captures the remaining repository-level tasks before making the cleaned tree public.

## Must Decide

- decide whether `paper/` should remain in the public repository as full manuscript source or move to a separate paper repo later

## Strongly Recommended

- verify the main public scripts run from a clean environment using `requirements.txt`
- review whether any kernel `.h5` files should stay out of GitHub and be distributed separately
- review whether any data files have external attribution or redistribution requirements that should be cited in `README.md`
- create the initial git repository and first commit from this cleaned directory

## Optional Cleanup

- remove local generated figures that are already ignored
- remove LaTeX build products in `paper/`
- add a `CITATION.cff` if you want GitHub citation metadata on the repository front page
