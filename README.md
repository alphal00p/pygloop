# pygloop
Python project steering gammaloop and spenso for various taylored custom applications in HEP

## Installation
To install pygloop from source and run example:

```bash
git clone git@github.com:alphal00p/pygloop.git pygloop
cd pygloop
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
git clone https://github.com/symbolica-dev/symbolica-community symbolica_community; 
cd symbolica_community
cargo run --features "python_stubgen" --no-default-features
maturin build --release # take note of wheel file created: <symbolica_community-wheel-file-path>
cd ..
python3 -m pip install <symbolica_community-wheel-file-path>
git clone -b hedge_numerator git@github.com:alphal00p/gammaloop.git gammaloop
cd gammaloop
maturin build -m gammaloop-api/Cargo.toml --features=ufo_support,python_api --profile=release # take note of wheel file created: <gammaloop-wheel-file-path>
python3 -m pip install <gammaloop-wheel-file-path>
```

PS: Currently tested with:
* `gammaLoop` `hedge_numerator` branch, with revhash `5c2ef4a0f803d0e70c6f8f87450e53ee427a2b9c`
* `symbolica_community` revhash `b5e57474329e94ca7544ad72315342e1f4e71a9c`
* `symbolica` dependency in `symbolica_community` from the `dev` branch and patched to revhash `4b472ae587bc0e354d7bfd12006230274bdf63fe`
* `spenso` dependencies of `symbolica_community` patched to revhash `ad0e8c24398ac14cce6b2b6b29dbc938e1833d8c`

## Examples
To run the examples:

```bash
cd pygloop
bash examples/gghhh.sh
```

## Tests

```bash
python3 -m pytest -m 'not_slow'
```

## Profiling

```bash
./examples/profile.py --n-loops 1 -r ./profiling_results/profiling_results.txt -t 10.0
```
