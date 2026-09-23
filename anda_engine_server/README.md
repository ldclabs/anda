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
- **Authentication**: ICP `SignedEnvelope` verification of the request body
  and optional bearer CWT tokens; unauthenticated callers
  are treated as anonymous and gated by each engine's visibility policy.
- **Pluggable middleware**: Compression, an optional `x-api-key` guard, and
  arbitrary axum/tower layers.

## Example

See the [anda_cli](../anda_cli/README.md) for example usage with this server.

## HTTP contract

| Endpoint | Response | Authentication and caching |
| --- | --- | --- |
| `GET /`, `GET /.well-known/information`, `GET /.well-known/agents` | `AppInformation`, including the resolved `caller` | Anonymous access is allowed; supplied credentials must verify. Responses use `Cache-Control: no-store`. |
| `GET /.well-known/agents/{id}` | Public `EngineCard` with exported functions | Does not authenticate or personalize the card; private functions are excluded. |
| `POST /{id}` | RPC result for `agent_run`, `tool_call`, or `information` | Supplied credentials must verify; the engine enforces execution permissions. |

`{id}` is an engine principal or `default`. Every registered map key must equal
the associated `Engine::id()`; `build_router()` rejects mismatches. An explicit
default must exist in that map; otherwise the first key in map order is used.

Discovery responses negotiate JSON/CBOR using `Accept` quality weights and
wildcards. A more specific range overrides a wildcard, including an explicit
`q=0` exclusion. JSON wins equal weights and is the default when `Accept` is
absent. If neither representation is acceptable, the server returns `406`.
Both representations are emitted without media parameters; ranges requiring
additional media parameters do not match. Discovery adds `Vary: Accept`, which
is retained alongside the compression layer's `Vary: Accept-Encoding`.

RPC uses the request `Content-Type` for both the outer envelope and the encoded
`params`/success payload: `application/cbor`, `application/json`, or a type with
a `+cbor`/`+json` suffix. Other types return `415`, and an undecodable envelope
returns `400`. `Accept` does not change the RPC codec. Execution and
parameter errors are returned inside the existing `RPCResponse::Err` envelope
with HTTP `200`; invalid credentials return `401`. The default axum request
body limit is 2 MiB and can be customized with a router middleware.

Middlewares are applied in registration order, with each `router.layer(...)`
wrapping the previous router: registering A then B runs requests as
`B → A → handler`, and responses as `handler → A → B`.
`with_app_name` and `with_app_version` configure `serve()` startup logs; use
`with_extra_info` to add discovery metadata.

## Authentication contract

`Bearer` and `ICP` scheme names are case-insensitive and accept one or more
spaces before the token. Token bytes are preserved. With trusted Ed25519 keys
configured, a bearer attempt must verify as a CWT with a valid principal in
`sub` and an `exp` claim. The CWT verifier also checks `nbf` when present and
allows five minutes of clock skew. Trusted keys are server-wide: bearer tokens
are not bound to the request body or a particular engine, and this server does
not enforce `aud` or scope claims. Engine access still depends on the resolved
principal's permissions.

Signed envelopes are read from `Authorization: ICP ...` or legacy `ic-auth-*`
headers. On RPC calls the client must provide an envelope digest equal to the
SHA3-256 hash of the received body. A malformed credential is rejected rather
than becoming anonymous. The public engine-card GET endpoint does not use
credentials, even when supplied; it returns no caller-specific information.

Direct-key signatures authenticate the body hash but **do not provide freshness
or automatic engine binding**. `expect_target` restricts only delegations that
carry target lists; `now_ms` is used for delegation expiry and canister-signature
certificates. Clients should set `RequestMeta::engine`, which both `agent_run`
and `tool_call` validate, and use TLS. A captured direct-key request can otherwise
be replayed, including against another engine where that principal has access.

Requiring an explicit digest does not add cryptographic domain separation: the
signature is still over the same body hash. Preventing replay or signatures
obtained from another use of the same key requires a coordinated protocol change
with freshness and domain binding across the signing library and clients.

## Architecture

The server is a stateless forwarder in front of the Anda engine, handling:
- HTTP request routing and CBOR/JSON codec negotiation
- Authentication (ICP signature verification and bearer CWT)
- Agent and tool dispatch into the target engine

Session management, tool integrations (shell, filesystem, fetch, etc.), and
per-caller access control are provided by [`anda_engine`](../anda_engine), not
by this crate.

RPC handlers release encoded parameter buffers after decoding, before awaiting
the engine. Server discovery borrows static metadata during serialization;
engine cards are rebuilt so refreshed dynamic tool definitions remain visible.
Skipping SHA3 for anonymous/CWT calls remains a profiling candidate; request
extraction and signature verification continue to use the existing body hash.

## License

Copyright © 2026 [LDC Labs](https://github.com/ldclabs).

`ldclabs/anda` is licensed under the MIT License. See the [MIT license][license] for the full license text.

[license]: ./../LICENSE-MIT
