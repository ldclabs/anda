use anda_core::Json;
use candid::Principal;
use chrono::prelude::*;
use rand::Rng;

pub mod context;
pub mod engine;
pub mod extension;
pub mod management;
pub mod memory;
pub mod model;
pub mod store;

/// Gets current unix timestamp in milliseconds
pub use structured_logger::unix_ms;

/// Generates N random bytes
pub use ic_cose::rand_bytes;

/// This is used to represent unauthenticated or anonymous users in the system.
pub const ANONYMOUS: Principal = Principal::anonymous();

pub static APP_USER_AGENT: &str = concat!(
    "Mozilla/5.0 anda.bot ",
    env!("CARGO_PKG_NAME"),
    "/",
    env!("CARGO_PKG_VERSION"),
);

/// Generates a random number within the given range
pub fn rand_number<T, R>(range: R) -> T
where
    T: rand::distr::uniform::SampleUniform,
    R: rand::distr::uniform::SampleRange<T>,
{
    let mut rng = rand::rng();
    rng.random_range(range)
}

/// Gets the current RFC 3339 datetime string
pub fn rfc3339_datetime_now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

/// Gets the RFC 3339 datetime string for the given timestamp in milliseconds
pub fn rfc3339_datetime(now_ms: u64) -> Option<String> {
    let datetime = DateTime::<Utc>::from_timestamp_millis(now_ms as i64);
    datetime.map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
}

/// Sets the Unix timestamp in milliseconds for each JSON object in the vector.
pub fn json_set_unix_ms_timestamp(mut vals: Vec<Json>, timestamp_ms: u64) -> Vec<Json> {
    for val in vals.iter_mut() {
        if let Some(obj) = val.as_object_mut() {
            obj.insert("timestamp".into(), timestamp_ms.into());
        }
    }
    vals
}

/// Converts the Unix timestamp in milliseconds to RFC 3339 format for each JSON object in the vector.
pub fn json_convert_rfc3339_timestamp(mut vals: Vec<Json>) -> Vec<Json> {
    for val in vals.iter_mut() {
        if let Some(obj) = val.as_object_mut()
            && let Some(timestamp_ms) = obj.get("timestamp").and_then(Json::as_u64)
        {
            obj.insert("timestamp".into(), rfc3339_datetime(timestamp_ms).into());
        }
    }
    vals
}
