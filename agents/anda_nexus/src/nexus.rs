use anda_core::{
    BoxError, FunctionDefinition, Json, Resource, ResourceRef, StateFeatures, Tool, ToolOutput,
    ToolSet, Xid, gen_schema_for, update_resources,
};
use anda_db::{
    collection::{Collection, CollectionConfig},
    database::AndaDB,
    error::DBError,
    index::BTree,
    query::{Filter, Query, RangeQuery},
};
use anda_db_schema::Fv;
use anda_db_tfs::jieba_tokenizer;
use anda_engine::{ANONYMOUS, context::BaseCtx, unix_ms};

use anda_kip::Response;
use candid::Principal;
use futures::stream::{self, StreamExt};
use parking_lot::RwLock;
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use crate::types::*;

#[derive(Debug)]
pub struct NexusNode {
    db: Arc<AndaDB>,
    threads: Arc<Collection>,
    thread_states: RwLock<BTreeMap<u64, Arc<RwLock<ThreadState>>>>,
}

impl NexusNode {
    fn thread_message_collection_name(id: u64) -> String {
        format!("{}_messages", id)
    }

    fn thread_resource_collection_name(id: u64) -> String {
        format!("{}_resources", id)
    }

    pub fn tools(nexus: Arc<NexusNode>) -> Result<ToolSet<BaseCtx>, BoxError> {
        let mut tools = ToolSet::new();
        tools.add(ThreadTool::new(nexus.clone()))?;
        tools.add(MessageTool::new(nexus.clone()))?;
        tools.add(GetResourceTool::new(nexus.clone()))?;
        Ok(tools)
    }

    pub async fn connect(db: Arc<AndaDB>) -> Result<Self, BoxError> {
        let schema = Thread::schema()?;
        let threads = db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: "threads".to_string(),
                    description: "threads collection".to_string(),
                },
                async |collection| {
                    // set tokenizer
                    collection.set_tokenizer(jieba_tokenizer());
                    // create BTree indexes if not exists
                    collection.create_btree_index_nx(&["id"]).await?;
                    collection.create_btree_index_nx(&["participants"]).await?;
                    collection
                        .create_bm25_index_nx(&["name", "tags", "description"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let thread_ids = threads.ids();

        let rt = stream::iter(thread_ids.into_iter())
            .map(|id| {
                let threads = threads.clone();

                async move {
                    let thread: Thread = match threads.get_as(id).await {
                        Ok(thread) => thread,
                        Err(DBError::NotFound { .. }) => return None,
                        Err(err) => return Some(Err(err)),
                    };

                    Some(Ok((thread._id, thread.to_state())))
                }
            })
            .buffer_unordered(16)
            .collect::<Vec<Option<Result<_, DBError>>>>()
            .await;

        let mut thread_states: BTreeMap<u64, Arc<RwLock<ThreadState>>> = BTreeMap::new();
        for r in rt.into_iter().flatten() {
            match r {
                Ok((id, state)) => {
                    thread_states.insert(id, Arc::new(RwLock::new(state)));
                }
                Err(err) => {
                    log::error!("Failed to load thread: {}", err);
                    continue;
                }
            }
        }

        Ok(Self {
            db,
            threads,
            thread_states: RwLock::new(thread_states),
        })
    }

    async fn get_message_collection(&self, thread_id: u64) -> Result<Arc<Collection>, BoxError> {
        let collection = self
            .db
            .open_collection(
                Self::thread_message_collection_name(thread_id),
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        let latest_message_id = {
            if let Some(state) = self.thread_states.read().get(&thread_id) {
                state.read().latest_message_id
            } else {
                0
            }
        };
        if latest_message_id == 0
            && let Some(id) = collection.latest_document_id()
            && let Ok(message) = collection.get_as::<Message>(id).await
            && let Some(state) = self.thread_states.write().get_mut(&thread_id)
        {
            let mut s = state.write();
            if message._id > s.latest_message_id {
                s.latest_message_by = message.user;
                s.latest_message_id = message._id;
                s.latest_message_at = message.timestamp;
            }
        }

        Ok(collection)
    }

    async fn get_resource_collection(&self, thread_id: u64) -> Result<Arc<Collection>, BoxError> {
        let collection = self
            .db
            .open_collection(
                Self::thread_resource_collection_name(thread_id),
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        Ok(collection)
    }

    pub async fn create_thread(
        &self,
        owner: Principal,
        name: String,
        description: Option<String>,
    ) -> Result<Thread, BoxError> {
        let updated_at = unix_ms();
        let mut thread = Thread {
            _id: 0,
            id: Xid::new(),
            name,
            controllers: BTreeSet::from([owner]),
            managers: BTreeSet::from([owner]),
            participants: BTreeMap::from([(owner, 0)]),
            created_at: updated_at,
            updated_at,
            description,
            ..Default::default()
        };
        let id = self.threads.add_from(&thread).await.unwrap();
        thread._id = id;

        let schema = Message::schema()?;
        let _ = self
            .db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: Self::thread_message_collection_name(id),
                    description: "Thread messages collection".to_string(),
                },
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());
                    collection.create_btree_index_nx(&["user"]).await?;
                    collection.create_bm25_index_nx(&["content"]).await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;
        let schema = Resource::schema()?;
        let _ = self
            .db
            .open_or_create_collection(
                schema,
                CollectionConfig {
                    name: Self::thread_resource_collection_name(id),
                    description: "Thread resources collection".to_string(),
                },
                async |collection| {
                    collection.set_tokenizer(jieba_tokenizer());
                    collection.create_btree_index_nx(&["tags"]).await?;
                    collection.create_btree_index_nx(&["hash"]).await?;
                    collection.create_btree_index_nx(&["mime_type"]).await?;
                    collection
                        .create_bm25_index_nx(&["name", "description", "metadata"])
                        .await?;

                    Ok::<(), DBError>(())
                },
            )
            .await?;

        self.thread_states
            .write()
            .insert(thread._id, Arc::new(RwLock::new(thread.to_state())));

        Ok(thread)
    }

    pub async fn fetch_my_threads_state(&self, user: &Principal) -> Vec<ThreadState> {
        let ids: Vec<u64> = self.my_thread_ids(user).await;
        let mut rt = Vec::with_capacity(ids.len());
        let states = self.thread_states.read();
        for id in ids {
            if let Some(s) = states.get(&id) {
                let s = s.read();
                if s.status == ThreadStatus::Active {
                    rt.push(s.clone());
                }
            }
        }
        rt
    }

    pub fn public_threads_state(&self, ids: BTreeSet<u64>) -> Vec<ThreadState> {
        let mut rt = Vec::with_capacity(ids.len());
        let states = self.thread_states.read();
        for id in ids {
            if let Some(s) = states.get(&id) {
                let s = s.read();
                if s.visibility == ThreadVisibility::Public && s.status == ThreadStatus::Active {
                    rt.push(s.clone());
                }
            }
        }
        rt
    }

    pub async fn get_thread(&self, user: &Principal, _id: u64) -> Result<Thread, BoxError> {
        self.check_thread_state(_id)?;

        let thread: Thread = self.threads.get_as(_id).await?;
        if thread.has_permission(user, ThreadPermission::Read) {
            Ok(thread)
        } else {
            Err(format!(
                "User {} does not have permission to access thread {}",
                user, _id
            )
            .into())
        }
    }

    pub async fn list_my_threads(
        &self,
        user: &Principal,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<ThreadInfo>, Option<String>), BoxError> {
        let limit = limit.unwrap_or(100).min(1000);
        let cursor = (BTree::from_cursor::<u64>(&cursor)?).unwrap_or_default();
        let mut ids: Vec<u64> = self.my_thread_ids(user).await;
        if cursor > 0 {
            ids.sort();
            ids.retain(|&id| id > cursor);
        }
        if ids.len() > limit {
            ids.truncate(limit);
        }
        let cursor = if ids.len() >= limit {
            BTree::to_cursor(&ids.last().unwrap())
        } else {
            None
        };

        let mut threads = Vec::with_capacity(ids.len());
        for id in ids {
            if let Ok(thread) = self.threads.get_as(id).await {
                threads.push(thread);
            }
        }
        Ok((threads, cursor))
    }

    pub async fn public_threads(&self, limit: Option<usize>) -> Result<Vec<ThreadInfo>, BoxError> {
        let limit = limit.unwrap_or(100).min(1000);

        let mut candidates = {
            let states = self.thread_states.read();
            let mut candidates = Vec::with_capacity(states.len() / 2);
            for (id, state) in states.iter() {
                let s = state.read();
                if (s.visibility == ThreadVisibility::Public
                    || s.visibility == ThreadVisibility::Protected)
                    && s.status == ThreadStatus::Active
                {
                    candidates.push((*id, s.updated_at));
                }
            }
            candidates
        };

        candidates.sort_by(|a, b| b.1.cmp(&a.1)); // sort by updated_at desc
        let mut threads = Vec::with_capacity(limit);
        for (id, _) in candidates.into_iter().take(limit) {
            if let Ok(thread) = self.threads.get_as(id).await {
                threads.push(thread);
            }
        }

        Ok(threads)
    }

    pub async fn update_thread(
        &self,
        user: &Principal,
        _id: u64,
        mut input: UpdateThreadInfo,
    ) -> Result<Thread, BoxError> {
        input.validate_and_normalize()?;
        self.check_thread_state(_id)?;

        let thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }
        let updated_at = unix_ms();
        let mut changes: BTreeMap<String, Fv> =
            BTreeMap::from([("updated_at".to_string(), Fv::U64(updated_at))]);
        if let Some(name) = input.name {
            changes.insert("name".to_string(), Fv::Text(name));
        }
        if let Some(language) = input.language {
            changes.insert("language".to_string(), Fv::Text(language));
        }
        if let Some(image) = input.image {
            changes.insert("image".to_string(), Fv::Text(image));
        }
        if let Some(tags) = input.tags {
            changes.insert(
                "tags".to_string(),
                Fv::Array(tags.into_iter().map(Fv::Text).collect()),
            );
        }
        if let Some(description) = input.description {
            changes.insert("description".to_string(), Fv::Text(description));
        }
        if let Some(visibility) = &input.visibility {
            changes.insert("visibility".to_string(), Fv::Text(visibility.to_string()));
        }

        let doc = self.threads.update(_id, changes).await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.updated_at = updated_at;
            if let Some(visibility) = input.visibility {
                s.visibility = visibility;
            }
        }
        Ok(doc.try_into()?)
    }

    pub async fn update_thread_controllers(
        &self,
        user: &Principal,
        _id: u64,
        controllers: BTreeSet<Principal>,
    ) -> Result<Thread, BoxError> {
        if controllers.is_empty() {
            return Err("Controllers cannot be empty".to_string().into());
        }
        if controllers.len() > 5 {
            return Err("Controllers cannot be more than 5".to_string().into());
        }
        self.check_thread_state(_id)?;

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        for p in &controllers {
            thread.participants.entry(*p).or_insert(0);
        }
        let controllers_fv = Fv::Array(
            controllers
                .into_iter()
                .map(|p| p.as_ref().to_vec().into())
                .collect(),
        );
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        let doc = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    ("controllers".to_string(), controllers_fv),
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(doc.try_into()?)
    }

    pub async fn update_thread_managers(
        &self,
        user: &Principal,
        _id: u64,
        managers: BTreeSet<Principal>,
    ) -> Result<Thread, BoxError> {
        if managers.is_empty() {
            return Err("Managers cannot be empty".to_string().into());
        }
        if managers.len() > 5 {
            return Err("Managers cannot be more than 5".to_string().into());
        }
        self.check_thread_state(_id)?;

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        for p in &managers {
            thread.participants.entry(*p).or_insert(0);
        }
        let managers_fv = Fv::Array(
            managers
                .into_iter()
                .map(|p| p.as_ref().to_vec().into())
                .collect(),
        );
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        let doc = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    ("managers".to_string(), managers_fv),
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(doc.try_into()?)
    }

    pub async fn add_thread_participants(
        &self,
        user: &Principal,
        _id: u64,
        participants: BTreeSet<Principal>,
    ) -> Result<Thread, BoxError> {
        if participants.is_empty() {
            return Err("Participants cannot be empty".to_string().into());
        }

        let (num_participants, max_participants) = {
            match self.thread_states.read().get(&_id) {
                Some(state) => {
                    let s = state.read();
                    if s.status != ThreadStatus::Active {
                        return Err(format!("Thread {} is not active", _id).into());
                    }
                    (s.participants, s.max_participants)
                }
                None => return Err(format!("Thread {} not found", _id).into()),
            }
        };

        if num_participants + participants.len() as u64 > max_participants {
            return Err(format!("Exceed max participants limit: {}", max_participants).into());
        }

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }

        for p in participants {
            thread.participants.entry(p).or_insert(0);
        }
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        let doc = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(doc.try_into()?)
    }

    pub async fn remove_thread_participants(
        &self,
        user: &Principal,
        _id: u64,
        participants: BTreeSet<Principal>,
    ) -> Result<Thread, BoxError> {
        if participants.is_empty() {
            return Err("Participants cannot be empty".to_string().into());
        }

        self.check_thread_state(_id)?;
        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Manage) {
            return Err(format!(
                "User {} does not have permission to manage thread {}",
                user, _id
            )
            .into());
        }

        if thread.controllers.intersection(&participants).count() > 0 {
            return Err("Cannot remove controllers from participants"
                .to_string()
                .into());
        }
        if thread.managers.intersection(&participants).count() > 0 {
            return Err("Cannot remove managers from participants"
                .to_string()
                .into());
        }

        for p in participants {
            thread.participants.remove(&p);
        }
        let updated_at = unix_ms();
        let participants = thread.participants.len() as u64;
        let doc = self
            .threads
            .update(
                _id,
                BTreeMap::from([
                    (
                        "participants".to_string(),
                        Fv::Map(
                            thread
                                .participants
                                .into_iter()
                                .map(|(k, v)| (k.as_ref().into(), v.into()))
                                .collect(),
                        ),
                    ),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(doc.try_into()?)
    }

    pub async fn quit_thread(&self, user: &Principal, _id: u64) -> Result<(), BoxError> {
        {
            match self.thread_states.read().get(&_id) {
                Some(state) => {
                    if state.read().status == ThreadStatus::Unavailable {
                        return Err(format!("Thread {} is unavailable", _id).into());
                    }
                }
                None => return Err(format!("Thread {} not found", _id).into()),
            }
        }

        let mut thread: Thread = self.threads.get_as(_id).await?;
        if !thread.participants.contains_key(user) {
            return Err(format!("User {} is not a participant of thread {}", user, _id).into());
        }

        let updated_at = unix_ms();
        let mut changes: BTreeMap<String, Fv> =
            BTreeMap::from([("updated_at".to_string(), Fv::U64(updated_at))]);
        if thread.controllers.contains(user) {
            thread.controllers.remove(user);
            if thread.controllers.is_empty() {
                return Err("Cannot quit thread as the last controller"
                    .to_string()
                    .into());
            }
            changes.insert(
                "controllers".to_string(),
                Fv::Array(
                    thread
                        .controllers
                        .into_iter()
                        .map(|p| p.as_ref().to_vec().into())
                        .collect(),
                ),
            );
        }

        if thread.managers.contains(user) {
            thread.managers.remove(user);
            changes.insert(
                "managers".to_string(),
                Fv::Array(
                    thread
                        .managers
                        .into_iter()
                        .map(|p| p.as_ref().to_vec().into())
                        .collect(),
                ),
            );
        }

        thread.participants.remove(user);
        let participants = thread.participants.len() as u64;
        changes.insert(
            "participants".to_string(),
            Fv::Map(
                thread
                    .participants
                    .into_iter()
                    .map(|(k, v)| (k.as_ref().into(), v.into()))
                    .collect(),
            ),
        );
        self.threads.update(_id, changes).await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.participants = participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn delete_thread(&self, user: &Principal, _id: u64) -> Result<(), BoxError> {
        {
            match self.thread_states.read().get(&_id) {
                Some(state) => {
                    if state.read().status == ThreadStatus::Unavailable {
                        return Err(format!("Thread {} is unavailable", _id).into());
                    }
                }
                None => return Err(format!("Thread {} not found", _id).into()),
            }
        }

        let thread: Thread = self.threads.get_as(_id).await?;
        if !thread.has_permission(user, ThreadPermission::Control) {
            return Err(format!(
                "User {} does not have permission to control thread {}",
                user, _id
            )
            .into());
        }

        {
            self.thread_states.write().remove(&_id);
        }

        self.threads.remove(_id).await?;
        self.db
            .delete_collection(Self::thread_resource_collection_name(_id).as_str())
            .await?;
        self.db
            .delete_collection(Self::thread_message_collection_name(_id).as_str())
            .await?;

        Ok(())
    }

    pub async fn sys_set_thread_status(
        &self,
        _id: u64,
        status: ThreadStatus,
    ) -> Result<(), BoxError> {
        let updated_at = unix_ms();
        self.threads
            .update(
                _id,
                BTreeMap::from([
                    ("status".to_string(), Fv::Text(status.to_string())),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.status = status;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    pub async fn sys_set_thread_max_participants(
        &self,
        _id: u64,
        max_participants: u64,
    ) -> Result<(), BoxError> {
        let updated_at = unix_ms();
        self.threads
            .update(
                _id,
                BTreeMap::from([
                    ("max_participants".to_string(), Fv::U64(max_participants)),
                    ("updated_at".to_string(), Fv::U64(updated_at)),
                ]),
            )
            .await?;
        if let Some(state) = self.thread_states.write().get_mut(&_id) {
            let mut s = state.write();
            s.max_participants = max_participants;
            s.updated_at = updated_at;
        }
        Ok(())
    }

    fn check_thread_state(&self, thread_id: u64) -> Result<ThreadVisibility, BoxError> {
        match self.thread_states.read().get(&thread_id) {
            Some(state) => {
                let s = state.read();
                if s.status != ThreadStatus::Active {
                    return Err(format!("Thread {} is not active", thread_id).into());
                }
                Ok(s.visibility)
            }
            None => Err(format!("Thread {} not found", thread_id).into()),
        }
    }

    async fn my_thread_ids(&self, user: &Principal) -> Vec<u64> {
        self.threads
            .search_ids(Query {
                filter: Some(Filter::Field((
                    "participants".to_string(),
                    RangeQuery::Eq(Fv::Bytes(user.as_slice().to_vec())),
                ))),
                ..Default::default()
            })
            .await
            .unwrap_or_default()
    }
}

impl NexusNode {
    pub async fn add_message(
        &self,
        user: &Principal,
        thread_id: u64,
        reply_to: u64,
        message: String,
        resources: Vec<Resource>,
    ) -> Result<Message, BoxError> {
        self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let collection = self.get_message_collection(thread_id).await?;
        let timestamp = unix_ms();
        let resources = update_resources(user, resources);
        let resources = self.try_add_resources(thread_id, &resources).await?;
        let content = vec![message.into()];
        let mut message = Message {
            _id: 0,
            role: "user".to_string(),
            user: Some(*user),
            content,
            resources,
            timestamp,
            reply_to,
        };

        if reply_to > 0 && !collection.contains(reply_to) {
            return Err(format!("Reply to message {} not found", reply_to).into());
        }

        let _id = collection.add_from(&message).await?;
        collection.flush(timestamp).await?;
        message._id = _id;

        if let Some(state) = self.thread_states.write().get_mut(&thread_id) {
            let mut s = state.write();
            s.latest_message_by = message.user;
            s.latest_message_id = message._id;
            s.latest_message_at = timestamp;
        }

        Ok(message)
    }

    pub async fn get_message(
        &self,
        user: &Principal,
        thread_id: u64,
        message_id: u64,
    ) -> Result<Message, BoxError> {
        let v = self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) && v != ThreadVisibility::Public {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let collection = self.get_message_collection(thread_id).await?;
        let message: Message = collection.get_as(message_id).await?;

        Ok(message)
    }

    pub async fn list_messages(
        &self,
        user: &Principal,
        thread_id: u64,
        cursor: Option<String>,
        limit: Option<usize>,
    ) -> Result<(Vec<Message>, Option<String>), BoxError> {
        let v = self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) && v != ThreadVisibility::Public {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let limit = limit.unwrap_or(100).min(1000);
        let cursor = (BTree::from_cursor::<u64>(&cursor)?).unwrap_or_default();

        let collection = self.get_message_collection(thread_id).await?;
        let mut message_ids = collection.ids();
        if cursor > 0 {
            message_ids.retain(|id| *id < cursor);
        }

        if message_ids.len() > limit {
            message_ids.drain(0..message_ids.len() - limit);
        }

        let mut messages = Vec::with_capacity(message_ids.len());
        for id in message_ids {
            if let Ok(message) = collection.get_as::<Message>(id).await {
                messages.push(message);
            }
        }
        let cursor = if messages.len() >= limit {
            BTree::to_cursor(&messages.first().unwrap()._id)
        } else {
            None
        };

        Ok((messages, cursor))
    }

    pub async fn delete_message(
        &self,
        user: &Principal,
        thread_id: u64,
        message_id: u64,
    ) -> Result<(), BoxError> {
        self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let timestamp = unix_ms();
        let collection = self.get_message_collection(thread_id).await?;
        if message_id != collection.latest_document_id().unwrap_or(0) {
            return Err("Can only delete the latest message".to_string().into());
        }

        let message: Message = collection.get_as(message_id).await?;
        if message.user != Some(*user) {
            return Err(format!(
                "User {} does not have permission to delete message {} in thread {}",
                user, message_id, thread_id
            )
            .into());
        }

        collection.remove(message_id).await?;
        collection.flush(timestamp).await?;
        let latest_message_id = collection.latest_document_id().unwrap_or_default();
        let (latest_message_by, latest_message_id, latest_message_at) =
            if let Ok(message) = collection.get_as::<Message>(latest_message_id).await {
                (message.user, message._id, message.timestamp)
            } else {
                (None, 0, 0)
            };

        // update latest message if needed
        if let Some(state) = self.thread_states.write().get_mut(&thread_id) {
            let mut s = state.write();
            s.latest_message_by = latest_message_by;
            s.latest_message_id = latest_message_id;
            s.latest_message_at = latest_message_at;
        }

        Ok(())
    }

    pub async fn get_resource(
        &self,
        user: &Principal,
        thread_id: u64,
        id: u64,
    ) -> Result<Resource, BoxError> {
        let v = self.check_thread_state(thread_id)?;
        let ids = self.my_thread_ids(user).await;
        if !ids.contains(&thread_id) && v != ThreadVisibility::Public {
            return Err(
                format!("User {} is not a participant of thread {}", user, thread_id).into(),
            );
        }

        let collection = self.get_resource_collection(thread_id).await?;
        let resource = collection.get_as(id).await?;
        Ok(resource)
    }

    async fn try_add_resources(
        &self,
        thread_id: u64,
        resources: &[Resource],
    ) -> Result<Vec<Resource>, BoxError> {
        let collection = self.get_resource_collection(thread_id).await?;
        let mut rs: Vec<Resource> = Vec::with_capacity(resources.len());
        let mut count = 0;
        for r in resources.iter() {
            let rf: ResourceRef = r.into();
            let id = if r._id > 0 {
                r._id // TODO: check if the resource exists and has permission
            } else {
                match collection.add_from(&rf).await {
                    Ok(id) => {
                        count += 1;
                        id
                    }
                    Err(DBError::AlreadyExists { _id, .. }) => _id,
                    Err(err) => Err(err)?,
                }
            };

            let r2 = Resource {
                _id: id,
                blob: None,
                ..r.clone()
            };
            rs.push(r2)
        }

        if count > 0 {
            let timestamp = unix_ms();
            collection.flush(timestamp).await?;
        }

        Ok(rs)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum ThreadToolArgs {
    /// Create a new thread
    Create {
        /// The name of the thread to create
        name: String,
        /// The description of the thread to create
        #[serde(skip_serializing_if = "Option::is_none")]
        description: Option<String>,
    },
    /// Update a thread basic info
    Update {
        /// The ID of the thread to update
        thread_id: u64,
        /// The info to update
        input: UpdateThreadInfo,
    },
    /// Update thread controllers
    UpdateControllers {
        /// The ID of the thread to update
        thread_id: u64,
        /// The new set of controllers
        #[schemars(schema_with = "principals_set_schema")]
        user_ids: BTreeSet<Principal>,
    },
    /// Update thread managers
    UpdateManagers {
        /// The ID of the thread to update
        thread_id: u64,
        /// The new set of managers
        #[schemars(schema_with = "principals_set_schema")]
        user_ids: BTreeSet<Principal>,
    },
    /// Add participants to a thread
    AddParticipants {
        /// The ID of the thread to update
        thread_id: u64,
        /// The user IDs to add as participants
        #[schemars(schema_with = "principals_set_schema")]
        user_ids: BTreeSet<Principal>,
    },
    /// Remove participants from a thread
    RemoveParticipants {
        /// The ID of the thread to update
        thread_id: u64,
        /// The user IDs to remove from participants
        #[schemars(schema_with = "principals_set_schema")]
        user_ids: BTreeSet<Principal>,
    },
    /// Quit from a thread
    Quit {
        /// The ID of the thread to quit
        thread_id: u64,
    },
    /// Delete a thread
    Delete {
        /// The ID of the thread to delete
        thread_id: u64,
    },
    /// Get a thread
    Get {
        /// The ID of the thread to get
        thread_id: u64,
    },
    /// List my threads
    ListMy {
        /// The cursor for pagination
        cursor: Option<String>,
        /// The limit for pagination, default to 100
        limit: Option<usize>,
    },
    /// List public threads
    ListPublic {
        /// The limit for pagination, default to 100
        limit: Option<usize>,
    },
    /// Fetch all my threads state
    FetchMyThreadsState {},
    /// Fetch specified public threads state
    FetchPublicThreadsState {
        /// The thread IDs to fetch
        thread_ids: Vec<u64>,
    },
}

/// A tool for conversation API
#[derive(Debug, Clone)]
pub struct ThreadTool {
    nexus: Arc<NexusNode>,
    schema: Json,
}

impl ThreadTool {
    pub const NAME: &'static str = "thread_api";

    /// Creates a new SearchConversationsTool instance
    pub fn new(nexus: Arc<NexusNode>) -> Self {
        let schema = gen_schema_for::<ThreadToolArgs>();
        Self { nexus, schema }
    }
}

impl Tool<BaseCtx> for ThreadTool {
    type Args = ThreadToolArgs;
    type Output = Response;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Anda Nexus thread API".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let caller = ctx.caller().to_owned();
        if caller == ANONYMOUS {
            return Err("unauthenticated".into());
        }

        let resp = match args {
            ThreadToolArgs::Create { name, description } => {
                let thread = self.nexus.create_thread(caller, name, description).await?;
                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::Update { thread_id, input } => {
                let thread = self.nexus.update_thread(&caller, thread_id, input).await?;
                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::UpdateControllers {
                thread_id,
                user_ids,
            } => {
                let thread = self
                    .nexus
                    .update_thread_controllers(&caller, thread_id, user_ids)
                    .await?;
                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::UpdateManagers {
                thread_id,
                user_ids,
            } => {
                let thread = self
                    .nexus
                    .update_thread_managers(&caller, thread_id, user_ids)
                    .await?;

                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::AddParticipants {
                thread_id,
                user_ids,
            } => {
                let thread = self
                    .nexus
                    .add_thread_participants(&caller, thread_id, user_ids)
                    .await?;

                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::RemoveParticipants {
                thread_id,
                user_ids,
            } => {
                let thread = self
                    .nexus
                    .remove_thread_participants(&caller, thread_id, user_ids)
                    .await?;

                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::Quit { thread_id } => {
                self.nexus.quit_thread(&caller, thread_id).await?;
                Response::Ok {
                    result: json!({ "quit": thread_id }),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::Delete { thread_id } => {
                self.nexus.delete_thread(&caller, thread_id).await?;
                Response::Ok {
                    result: json!({ "deleted": thread_id }),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::Get { thread_id } => {
                let thread = self.nexus.get_thread(&caller, thread_id).await?;
                Response::Ok {
                    result: json!(thread),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::ListMy { cursor, limit } => {
                let (threads, next_cursor) =
                    self.nexus.list_my_threads(&caller, cursor, limit).await?;
                Response::Ok {
                    result: json!(threads),
                    next_cursor,
                }
            }
            ThreadToolArgs::ListPublic { limit } => {
                let threads = self.nexus.public_threads(limit).await?;
                Response::Ok {
                    result: json!(threads),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::FetchMyThreadsState {} => {
                let states = self.nexus.fetch_my_threads_state(&caller).await;
                Response::Ok {
                    result: json!(states),
                    next_cursor: None,
                }
            }
            ThreadToolArgs::FetchPublicThreadsState { thread_ids } => {
                let ids: BTreeSet<u64> = thread_ids.into_iter().collect();
                let states = self.nexus.public_threads_state(ids);
                Response::Ok {
                    result: json!(states),
                    next_cursor: None,
                }
            }
        };

        Ok(ToolOutput::new(resp))
    }
}

// ...existing code...

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum MessageToolArgs {
    /// Add a message to a thread
    Add {
        /// Thread ID
        thread_id: u64,
        /// The message
        message: String,
        /// Reply to message ID
        reply_to: Option<u64>,
    },
    /// Get a message in a thread
    Get {
        /// Thread ID
        thread_id: u64,
        /// Message ID
        message_id: u64,
    },
    /// List messages in a thread (倒序分页：cursor 为上一页最早消息的 _id)
    List {
        thread_id: u64,
        cursor: Option<String>,
        /// default 100, max 1000
        limit: Option<usize>,
    },
    /// Delete the latest message (只能删除最新一条且必须本人)
    Delete { thread_id: u64, message_id: u64 },
}

/// A tool for thread messages API
#[derive(Debug, Clone)]
pub struct MessageTool {
    nexus: Arc<NexusNode>,
    schema: Json,
}

impl MessageTool {
    pub const NAME: &'static str = "message_api";

    pub fn new(nexus: Arc<NexusNode>) -> Self {
        let schema = gen_schema_for::<MessageToolArgs>();
        Self { nexus, schema }
    }
}

impl Tool<BaseCtx> for MessageTool {
    type Args = MessageToolArgs;
    type Output = Response;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Anda Nexus message API".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let caller = ctx.caller().to_owned();
        if caller == ANONYMOUS {
            return Err("unauthenticated".into());
        }

        let resp = match args {
            MessageToolArgs::Add {
                thread_id,
                message,
                reply_to,
            } => {
                let msg = self
                    .nexus
                    .add_message(
                        &caller,
                        thread_id,
                        reply_to.unwrap_or_default(),
                        message,
                        resources,
                    )
                    .await?;
                Response::Ok {
                    result: json!(msg),
                    next_cursor: None,
                }
            }
            MessageToolArgs::Get {
                thread_id,
                message_id,
            } => {
                let msg = self
                    .nexus
                    .get_message(&caller, thread_id, message_id)
                    .await?;
                Response::Ok {
                    result: json!(msg),
                    next_cursor: None,
                }
            }
            MessageToolArgs::List {
                thread_id,
                cursor,
                limit,
            } => {
                let (messages, next_cursor) = self
                    .nexus
                    .list_messages(&caller, thread_id, cursor, limit)
                    .await?;
                Response::Ok {
                    result: json!(messages),
                    next_cursor,
                }
            }
            MessageToolArgs::Delete {
                thread_id,
                message_id,
            } => {
                self.nexus
                    .delete_message(&caller, thread_id, message_id)
                    .await?;
                Response::Ok {
                    result: json!({ "deleted": message_id }),
                    next_cursor: None,
                }
            }
        };

        Ok(ToolOutput::new(resp))
    }
}

/// Get a resource in a thread
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "camelCase")]
pub struct GetResourceToolArgs {
    /// Thread ID
    thread_id: u64,
    /// Resource ID
    resource_id: u64,
}

/// A tool for thread messages API
#[derive(Debug, Clone)]
pub struct GetResourceTool {
    nexus: Arc<NexusNode>,
    schema: Json,
}

impl GetResourceTool {
    pub const NAME: &'static str = "get_resource_api";

    pub fn new(nexus: Arc<NexusNode>) -> Self {
        let schema = gen_schema_for::<GetResourceToolArgs>();
        Self { nexus, schema }
    }
}

impl Tool<BaseCtx> for GetResourceTool {
    type Args = GetResourceToolArgs;
    type Output = Resource;

    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    fn description(&self) -> String {
        "Get Resource API".to_string()
    }

    fn definition(&self) -> FunctionDefinition {
        FunctionDefinition {
            name: self.name(),
            description: self.description(),
            parameters: self.schema.clone(),
            strict: Some(true),
        }
    }

    async fn call(
        &self,
        ctx: BaseCtx,
        args: Self::Args,
        _resources: Vec<Resource>,
    ) -> Result<ToolOutput<Self::Output>, BoxError> {
        let caller = ctx.caller().to_owned();
        if caller == ANONYMOUS {
            return Err("unauthenticated".into());
        }

        let res = self
            .nexus
            .get_resource(&caller, args.thread_id, args.resource_id)
            .await?;

        Ok(ToolOutput::new(res))
    }
}

fn principals_set_schema(generator: &mut SchemaGenerator) -> Schema {
    Vec::<String>::json_schema(generator)
}
