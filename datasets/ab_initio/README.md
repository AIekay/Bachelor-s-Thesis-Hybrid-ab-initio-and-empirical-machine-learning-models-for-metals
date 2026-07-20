# Ab initio dataset

## Source dataset 

- Dataset download: <https://doi.org/10.5281/zenodo.4734036>

```bibtex
@article{lysogorskiy2021performant_Cu_dataset,
  author  = {Lysogorskiy, Yury and van der Oord, Cas and Bochkarev, Anton and Menon, Sarath and Rinaldi, Matteo and Hammerschmidt, Thomas and Mrovec, Matous and Thompson, Aidan and Cs{\'a}nyi, G{\'a}bor and Ortner, Christoph and Drautz, Ralf},
  title   = {Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon},
  journal = {npj Computational Materials},
  volume  = {7},
  pages   = {97},
  year    = {2021},
  doi     = {10.1038/s41524-021-00559-9}
}

@dataset{Cu_dataset_download_lysogorskiy_2021_4734036,
  author       = {Lysogorskiy, Yury},
  title        = {{Performant implementation of the atomic cluster
                   expansion (PACE): Application to copper and
                   silicon}},
  month        = may,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.4734036},
  url          = {https://doi.org/10.5281/zenodo.4734036}
}
```

Format: JSON list of records, each with per-configuration coordinates, atom count, energy
(+ per-atom corrected energy), forces, cell/PBC info.

## Derived training data

`Cu_DATA.zip` — the same configurations re-exported into DeePMD-kit raw/set format (see
[`../README.md`](../README.md)), one folder per system under `pbc/` (`Cu10`, `Cu105`, ...,
`Cu1_expanded`, `Cu2_expanded`). This project's own re-processing, not on Zenodo. Unzip here
(`Cu_DATA/pbc/...`) before pointing `<DATA_ROOT>` at this folder.

`Cu1_expanded`/`Cu2_expanded` are 1-/2-atom primitive cells supercell-replicated for the cutoff
radius — real per-frame atom count is larger than the name suggests.
