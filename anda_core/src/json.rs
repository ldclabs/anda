use schemars::{JsonSchema, Schema, generate::SchemaSettings, transform::RestrictFormats};
use serde_json::{Map, Value, json};

/// Generates a JSON Schema document for `T`.
///
/// The schema is generated with draft 2020-12 settings, inline subschemas, no
/// `$schema` field, and restricted format inference. This keeps function-call
/// schemas compact and provider-friendly.
pub fn root_schema_for<T: JsonSchema>() -> Schema {
    let settings = SchemaSettings::draft2020_12().with(|s| {
        s.inline_subschemas = true;
        s.meta_schema = None; // Remove the $schema field

        let mut formater = RestrictFormats::default();
        formater.infer_from_meta_schema = false; // Do not infer formats from meta schema
        s.transforms.push(Box::new(formater)); // Remove the $format field
    });
    let generator = settings.into_generator();
    generator.into_root_schema_for::<T>()
}

/// Generates a compact, strict JSON Schema value for `T`.
///
/// Top-level `title` and `description` fields are removed. Object schemas are
/// normalized so `required` contains every key in `properties`, which matches
/// the strict function schema accepted by providers such as OpenAI.
pub fn gen_schema_for<T: JsonSchema>() -> serde_json::Value {
    let mut schema = root_schema_for::<T>();
    schema.remove("title");
    schema.remove("description");
    normalize_strict_schema(schema.to_value())
}

/// Normalizes a JSON Schema for strict function calling.
///
/// For every object schema, `additionalProperties` defaults to `false` and
/// `required` is rewritten to contain all property keys. Object schemas without
/// explicit properties are normalized to empty closed objects.
pub fn normalize_strict_schema(mut schema: Value) -> Value {
    normalize_schema_value(&mut schema);
    schema
}

fn normalize_schema_value(schema: &mut Value) {
    match schema {
        Value::Object(map) => normalize_schema_object(map),
        Value::Array(items) => {
            for item in items {
                normalize_schema_value(item);
            }
        }
        _ => {}
    }
}

fn normalize_schema_object(map: &mut Map<String, Value>) {
    let is_object = schema_type_contains_object(map.get("type"));

    if is_object && !map.contains_key("properties") {
        map.insert("properties".to_string(), json!({}));
    }

    if is_object {
        map.entry("additionalProperties".to_string())
            .or_insert(Value::Bool(false));
    }

    if let Some(Value::Object(properties)) = map.get("properties") {
        let required = properties.keys().cloned().map(Value::String).collect();
        map.insert("required".to_string(), Value::Array(required));
    }

    for key in ["properties", "$defs", "definitions", "patternProperties"] {
        if let Some(Value::Object(children)) = map.get_mut(key) {
            for child in children.values_mut() {
                normalize_schema_value(child);
            }
        }
    }

    for key in ["items", "additionalProperties", "not", "if", "then", "else"] {
        if let Some(child) = map.get_mut(key)
            && child.is_object()
        {
            normalize_schema_value(child);
        }
    }

    for key in ["allOf", "anyOf", "oneOf", "prefixItems"] {
        if let Some(Value::Array(children)) = map.get_mut(key) {
            for child in children {
                normalize_schema_value(child);
            }
        }
    }
}

fn schema_type_contains_object(value: Option<&Value>) -> bool {
    match value {
        Some(Value::String(value)) => value == "object",
        Some(Value::Array(values)) => values
            .iter()
            .any(|value| value.as_str().is_some_and(|value| value == "object")),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
    struct TestStruct {
        name: String,
        age: Option<u8>,
    }

    #[test]
    fn test_root_schema_for() {
        let schema = gen_schema_for::<TestStruct>();
        let s = serde_json::to_string(&schema).unwrap();
        println!("{}", s);
        assert_eq!(
            schema,
            serde_json::json!({"type":"object","properties":{"name":{"type":"string"},"age":{"type":["integer","null"],"maximum":255,"minimum":0}},"required":["name","age"],"additionalProperties":false})
        );
    }

    #[test]
    fn test_normalize_strict_schema_recurses_into_nested_objects() {
        let schema = normalize_strict_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": { "type": "string" },
                            "enabled": { "type": "boolean" }
                        },
                        "required": ["id"]
                    }
                },
                "mode": { "type": "string" }
            },
            "required": ["items"]
        }));

        assert_eq!(schema["required"], serde_json::json!(["items", "mode"]));
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["items"]["items"]["required"],
            serde_json::json!(["id", "enabled"])
        );
        assert_eq!(
            schema["properties"]["items"]["items"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn test_normalize_strict_schema_handles_nullable_objects() {
        let schema = normalize_strict_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "maybe": {
                    "type": ["object", "null"],
                    "properties": {
                        "id": { "type": "string" }
                    }
                },
                "empty": {
                    "type": ["object", "null"]
                }
            }
        }));

        assert_eq!(
            schema["properties"]["maybe"]["required"],
            serde_json::json!(["id"])
        );
        assert_eq!(schema["properties"]["maybe"]["additionalProperties"], false);
        assert_eq!(schema["properties"]["empty"]["additionalProperties"], false);
        assert_eq!(
            schema["properties"]["empty"]["properties"],
            serde_json::json!({})
        );
        assert_eq!(
            schema["properties"]["empty"]["required"],
            serde_json::json!([])
        );
    }

    #[test]
    fn test_normalize_strict_schema_closes_propertyless_nested_objects() {
        let schema = normalize_strict_schema(serde_json::json!({
            "type": "object",
            "properties": {
                "commands": {
                    "type": "array",
                    "items": { "type": "string" }
                },
                "parameters": {
                    "type": "object",
                    "description": "An optional JSON object."
                }
            },
            "required": ["commands"]
        }));

        assert_eq!(
            schema["properties"]["parameters"]["properties"],
            serde_json::json!({})
        );
        assert_eq!(
            schema["properties"]["parameters"]["required"],
            serde_json::json!([])
        );
        assert_eq!(
            schema["properties"]["parameters"]["additionalProperties"],
            false
        );
    }
}
