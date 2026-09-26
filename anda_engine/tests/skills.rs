//! The skill bundle dispatches through ordinary AgentCtx tools without a model or shell.
use anda_core::{AgentContext, BoxError, ToolInput};
use anda_engine::{engine::EngineBuilder, extension::skill::SkillManager};
use serde_json::json;
use std::{fs, sync::Arc};

#[tokio::test]
async fn skill_resource_dispatch_rechecks_filter_and_validates_schema() -> Result<(), BoxError> {
    let root = std::env::temp_dir().join(format!(
        "anda-skill-dispatch-{:016x}",
        rand::random::<u64>()
    ));
    fs::create_dir_all(root.join("example/references"))?;
    fs::write(
        root.join("example/SKILL.md"),
        "---\nname: example\ndescription: Example instructions.\n---\nRead references/guide.md.\n",
    )?;
    fs::write(
        root.join("example/references/guide.md"),
        "Read this complete reference.",
    )?;
    let result = async {
        let manager = Arc::new(SkillManager::new(root.clone()));
        manager.load().await?;
        let ctx = EngineBuilder::new()
            .register_tools(manager.tools()?)?
            .mock_ctx();
        let call = |name: &str, args| ToolInput {
            name: name.into(),
            args,
            resources: vec![],
            ..Default::default()
        };
        let (listed, _) = ctx
            .tool_call(call("skills_list", json!({"query":null,"cursor":null})))
            .await?;
        let id = listed.output["skills"][0]["id"].as_str().unwrap();
        let (read, _) = ctx
            .tool_call(call(
                "skills_read",
                json!({"skill":id,"resource":"references/guide.md","cursor":null}),
            ))
            .await?;
        assert_eq!(read.output["content"], "Read this complete reference.");
        assert!(read.output["next_cursor"].is_null());
        assert!(
            ctx.tool_call(call(
                "skills_read",
                json!({"skill":id,"resource":"../SKILL.md","cursor":null})
            ))
            .await
            .is_err()
        );
        assert!(
            ctx.tool_call(call("skills_read", json!({"name":"example"})))
                .await
                .is_err()
        );
        manager.set_skill_filter(Some(Arc::new(|_| false)));
        assert!(
            ctx.tool_call(call(
                "skills_read",
                json!({"skill":id,"resource":null,"cursor":null})
            ))
            .await
            .is_err()
        );
        Ok::<_, BoxError>(())
    }
    .await;
    fs::remove_dir_all(root)?;
    result
}
