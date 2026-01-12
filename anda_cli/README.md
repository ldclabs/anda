# `anda_cli`

## Running locally

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
cp example.env .env
# update .env
cargo build -p anda_cli

./target/debug/anda_cli --help
./target/debug/anda_cli rand-bytes -l 48 -f hex
./target/debug/anda_cli agent-run --help
./target/debug/anda_cli agent-run -p 'Please check my PANDA balance'
./target/debug/anda_cli agent-run --id path_to_my_identity.pem -p 'Please check my PANDA balance'
```

## License
Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
