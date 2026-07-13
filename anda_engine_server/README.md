# `anda_engine_server`

A thin, stateless HTTP front-end that exposes one or more Anda engines over a
signed CBOR/JSON RPC endpoint plus `.well-known` discovery routes. Agent logic,
tools, and any session/persistence state live in the [`anda_engine`](../anda_engine)
runtime; this crate only authenticates callers and forwards typed requests to
the target engine.

## Features

- **Multi-Engine Support**: Serve multiple Anda engine instances behind one
  server, addressed by engine principal or the `default` alias.
- **Signed RPC**: `POST /{id}` dispatches `agent_run` / `tool_call` / `information`,
  with CBOR or JSON negotiated from the request `Content-Type`.
- **Discovery**: `.well-known` routes advertise each engine's public
  `EngineCard` (only exported agents/tools) to anonymous callers.
- **Authentication**: ICP `SignedEnvelope` verification (request body bound to
  the engine principal) and optional bearer CWT tokens; unauthenticated callers
  are treated as anonymous and gated by each engine's visibility policy.
- **Pluggable middleware**: Compression, an optional `x-api-key` guard, and
  arbitrary axum/tower layers.

## Example

See the [anda_cli](../anda_cli/README.md) for example usage with this server.

## Architecture

The server is a stateless forwarder in front of the Anda engine, handling:
- HTTP request routing and CBOR/JSON codec negotiation
- Authentication (ICP signature verification and bearer CWT)
- Agent and tool dispatch into the target engine

Session management, tool integrations (shell, filesystem, fetch, etc.), and
per-caller access control are provided by [`anda_engine`](../anda_engine), not
by this crate.

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
