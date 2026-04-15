# `anda_engine_server`

An HTTP server that serves multiple Anda engines, providing a REST API for agent execution and tool invocation.

## Features

- **Multi-Engine Support**: Run multiple Anda engine instances on a single server
- **HTTP API**: JSON-based API for agent execution
- **Tool Integration**: Access to built-in tools (shell, filesystem, fetch, etc.)
- **Session Management**: Support for persistent agent sessions

## Example

See the [anda_cli](../anda_cli/README.md) for example usage with this server.

## Architecture

The server acts as a front-end for the Anda engine, handling:
- HTTP request routing
- Session management
- Authentication (via ICP signature verification)
- Tool and agent dispatch

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
