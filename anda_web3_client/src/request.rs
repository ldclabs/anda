use anda_core::{BoxError, RPCRequestRef};
use ic_auth_types::ByteBufB64;

/// Wraps already-encoded params without decoding or re-encoding their contents.
pub(crate) fn rpc_body(method: &str, params: Vec<u8>) -> Result<Vec<u8>, BoxError> {
    let params = ByteBufB64::from(params);
    Ok(cbor2::to_canonical_vec(&RPCRequestRef {
        method,
        params: &params,
    })?)
}

#[cfg(feature = "client")]
/// Validates that `url` is a well-formed HTTP(S) endpoint this client may call.
///
/// Only the `https` scheme is accepted by default. The `http` scheme is
/// additionally allowed when [`crate::client::ClientBuilder::with_allow_http`] was enabled
/// (intended for local development against a replica or test server). Every
/// other scheme (`file`, `ftp`, `data`, ...) is rejected, and the URL must
/// carry a host.
///
/// Embedded userinfo is rejected: in `https://api.trusted.example@evil.tld/`
/// the authority is `evil.tld`, but the text reads as the trusted host. That
/// mismatch is the classic URL-smuggling primitive, and here it would send a
/// request signed with the client identity to the attacker's host.
///
/// This is a syntactic guard, not an SSRF firewall: it does not block
/// private, loopback, or link-local hosts (e.g. cloud metadata at
/// `169.254.169.254`), and it does not re-validate redirect hops. When this
/// client is used as a library, treat every endpoint passed to a signed call
/// as trusted — the request is signed with the client identity before it is
/// sent, so an attacker-controlled endpoint receives a valid signed request.
pub(crate) fn check_url(url: &str, allow_http: bool) -> Result<reqwest::Url, BoxError> {
    let parsed = reqwest::Url::parse(url).map_err(|err| format!("Invalid url {url:?}: {err}"))?;
    let scheme = parsed.scheme();
    let scheme_ok = scheme == "https" || (allow_http && scheme == "http");
    if !scheme_ok {
        let expected = if allow_http { "http or https" } else { "https" };
        return Err(
            format!("Invalid url {url:?}: scheme must be {expected}, got {scheme:?}").into(),
        );
    }
    if !parsed.has_host() {
        return Err(format!("Invalid url {url:?}: missing host").into());
    }
    if !parsed.username().is_empty() || parsed.password().is_some() {
        return Err(format!("Invalid url {url:?}: embedded userinfo is not allowed").into());
    }
    Ok(parsed)
}

#[cfg(feature = "client")]
/// Sends an HTTP request with optional headers and body using the given client.
pub(crate) async fn send_request(
    http: reqwest::Client,
    url: reqwest::Url,
    method: http::Method,
    headers: Option<http::HeaderMap>,
    body: Option<Vec<u8>>,
) -> Result<reqwest::Response, BoxError> {
    let mut req = http.request(method, url);
    if let Some(headers) = headers {
        req = req.headers(headers);
    }
    if let Some(body) = body {
        req = req.body(body);
    }

    req.send().await.map_err(|e| e.into())
}
