# From Rules to Rows: Exact Constrained Decoding for LLM-Based Tabular Synthesis

Install with Python 3.11 or 3.12:

```sh
pip install -e .
```

Fine-tune GReaT with distilgpt2 and compare Native with +TabCD:

```sh
python run.py --dataset syn-affine
python run.py --dataset computer-hardware
```

Use `--epochs 0.1 --n_rows 4` for a smaller run. CUDA is selected when available;
otherwise the runner uses CPU. Generated tables are saved in `outputs/`.

| Directory | Contents |
| --- | --- |
| `src/cdtab/` | Rule compiler, exact projections, token masks, reuse, GReaT bridge |
| `configs/rules/` | Syn-Affine and Computer Hardware constraints |
| `data/` | Synthetic data generator and [UCI Computer Hardware](https://archive.ics.uci.edu/dataset/29/computer+hardware) downloader |

Run core tests with `python -m unittest discover -s tests -v`.

MIT License; see [LICENSE](LICENSE).
