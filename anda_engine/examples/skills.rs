//! Load a local skill catalog and register its bounded tools without filesystem/shell grants.
//! Run with: cargo run -p anda_engine --example skills -- /absolute/path/to/skills
use anda_core::{BoxError, Tool};
use anda_engine::{
    engine::EngineBuilder,
    extension::skill::{SkillManager, SkillsListArgs, SkillsListTool},
};
use std::{path::PathBuf, sync::Arc};

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("Pass a skills directory")?;
    let manager = Arc::new(SkillManager::new(root));
    let report = manager.reload().await?;
    let ctx = EngineBuilder::new()
        .register_tools(manager.tools()?)?
        .mock_ctx();
    let page = SkillsListTool::new(manager)
        .call(ctx.base, SkillsListArgs::default(), vec![])
        .await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    println!("{}", serde_json::to_string_pretty(&page.output)?);
    Ok(())
}
