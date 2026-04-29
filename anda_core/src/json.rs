use schemars::{JsonSchema, Schema, generate::SchemaSettings, transform::RestrictFormats};

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

/// Generates a compact JSON Schema value for `T`.
///
/// Top-level `title` and `description` fields are removed. A missing
/// `required` array is normalized to an empty array.
pub fn gen_schema_for<T: JsonSchema>() -> serde_json::Value {
    let mut schema = root_schema_for::<T>();
    schema.remove("title");
    schema.remove("description");
    if schema.get("required").is_none() {
        schema.insert("required".to_string(), serde_json::json!([]));
    }
    schema.to_value()
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
            serde_json::json!({"type":"object","properties":{"age":{"type":["integer","null"],"maximum":255,"minimum":0},"name":{"type":"string"}},"required":["name"]})
        );
    }
}
