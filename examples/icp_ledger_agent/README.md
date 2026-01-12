# `icp_ledger_agent`

An AI agent example to interact with ICP blockchain ledgers.

## Running locally

```sh
git clone https://github.com/ldclabs/anda.git
cd anda
cp example.env .env
# update .env
cargo run -p icp_ledger_agent
```

Install CLI tool and run agent:
```sh
cargo install anda_cli
anda_cli agent-run -p 'Please check my PANDA balance'
anda_cli tool-call -n icp_ledger_balance_of -a '{"account":"535yc-uxytb-gfk7h-tny7p-vjkoe-i4krp-3qmcl-uqfgr-cpgej-yqtjq-rqe","symbol":"PANDA"}'
```

## License
Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../../LICENSE-MIT
