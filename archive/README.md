# Coursework source archive

These snapshots preserve the cell source text of the supplied notebooks, including course prompts, provided scaffolding, and student implementations. Saved outputs, execution counts, and runtime metadata were removed to keep the archive readable and to avoid mixing historical results with current reruns. Original full-upload hashes and cell-source hashes are recorded in [provenance.json](../docs/provenance.json).

| Snapshot | Disposition |
| --- | --- |
| [Bank marketing](bank-marketing/original_source.ipynb) | Reproduced and extended in the featured bank case; original report retained separately |
| [Donor targeting](donor-routing/original_source.ipynb) | Reproduced and refactored in the featured routing case |
| [Hotel overbooking](hotel-overbooking/original_source.ipynb) | Reviewed; feature-row alignment requires correction before a fresh evaluation |
| [Inventory optimization](inventory-optimization/original_source.ipynb) | Reviewed; raw sales file missing from this supplied batch, and static features need a training-only construction |
| [Causal targeting](causal-targeting/original_source.ipynb) | Reviewed; customer-level splitting, observation windows, and identification require revision |

The two supplied causal-targeting notebooks have identical cell sources. Only the executed source version is retained. The midterm ZIP's empty template is omitted because the completed notebook supplies the actual implementation.

Archive notebooks are provenance material. Their source may depend on unavailable input files, optional libraries, or the original course environment. They are not covered by the featured experiments' end-to-end reproduction claim. See [case review](../docs/CASE_REVIEW.md).
