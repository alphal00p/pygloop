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
cd ..
cd pygloop
bash examples/gghhh.sh
```
