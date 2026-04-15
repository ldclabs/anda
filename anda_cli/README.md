# `anda_cli`

Command-line interface for interacting with the Anda engine server.

## Features

- **Agent Execution**: Run AI agents with natural language prompts
- **Cryptographic Utilities**: Generate random bytes for key generation
- **Identity Management**: Support for PEM-based identity files

## Installation

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
cp example.env .env
# Update .env with your configuration
cargo build -p anda_cli
```

## Usage

```sh
# Show help
./target/debug/anda_cli --help

# Generate random bytes
./target/debug/anda_cli rand-bytes -l 48 -f hex

# Run an agent
./target/debug/anda_cli agent-run --help
./target/debug/anda_cli agent-run -p 'Please check my PANDA balance'
./target/debug/anda_cli agent-run --id path_to_my_identity.pem -p 'Please check my PANDA balance'
```

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
