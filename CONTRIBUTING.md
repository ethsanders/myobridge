# Contributing

Thanks for looking. Bug reports, recordings from your own hardware and pull
requests are all welcome.

## Set up

```bash
git clone https://github.com/ethsanders/myobridge
cd myobridge
pip install -e ".[dev]"
pytest
```

The tests run in about a minute without any downloads: everything that
needs a dataset uses small fake recordings in the real file formats. The
firmware tests compile the C++ headers with `g++` when it is installed.

Before opening a pull request:

```bash
ruff check . && ruff format --check .
mypy
pytest
```

CI runs the same on Python 3.10 to 3.13 and compiles the ESP32 sketch.

## What helps most

- **Recordings from your own armband.** A few minutes per gesture from
  `myobridge record`, with a note on the sensors and placement, is worth
  more than any code change right now. Open an issue to discuss sharing them.
- **Bugs with a reproduction.** The output of `myobridge listen` or a short
  script is ideal.
- **New datasets.** A loader in `src/myobridge/datasets/` that pins a public
  dataset to a commit or DOI, with a fake-data test like the existing ones.
- **Integrations.** Lab Streaming Layer output, a Unity package, a LibEMG
  adapter: open an issue first so we can agree on the shape.

## Ground rules for results

Numbers in the docs are produced by `myobridge bench` and checked by
`tests/test_docs.py`. If a change moves a number, rerun the benchmark it
belongs to and update the docs in the same pull request. Settings are chosen
on validation people or days and reported on held-out ones; please keep it
that way.

## Docs

The site is built with MkDocs Material from `docs/`:

```bash
pip install -e ".[docs]"
python scripts/build_demo.py   # the wheel the browser demo installs
mkdocs serve
```

## Code of conduct

Be kind and assume good faith. Harassment of any kind isn't welcome here.
