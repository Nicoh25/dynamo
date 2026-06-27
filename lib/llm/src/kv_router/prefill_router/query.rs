// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use anyhow::Result;
use dynamo_kv_router::protocols::{BlockExtraInfo, RoutingConstraints, WorkerId};

use super::{
    InnerPrefillRouter, PrefillError, PrefillLifecycleState, PrefillQueryOutcome, PrefillRouter,
};

impl PrefillRouter {
    /// Query the best prefill worker without executing a request.
    ///
    /// This query is advisory and does not book scheduler or occupancy state;
    /// concurrent callers may observe the same worker.
    #[expect(clippy::too_many_arguments)]
    pub async fn query_prefill_worker(
        &self,
        token_ids: &[u32],
        block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
        lora_name: Option<String>,
        priority_jump: f64,
        strict_priority: u32,
        allowed_worker_ids: Option<HashSet<WorkerId>>,
        routing_constraints: RoutingConstraints,
    ) -> Result<PrefillQueryOutcome> {
        if self.lifecycle_state() != PrefillLifecycleState::Active {
            return Err(anyhow::anyhow!(PrefillError::NotActivated));
        }
        let prefill_router = self
            .prefill_router
            .get()
            .ok_or_else(|| anyhow::anyhow!(PrefillError::NotActivated))?;

        match prefill_router {
            InnerPrefillRouter::KvRouter(router) => {
                let outcome = router
                    .chooser
                    .find_best_match_details(
                        None,
                        token_ids,
                        block_mm_infos,
                        None,
                        false,
                        false,
                        lora_name,
                        priority_jump,
                        strict_priority,
                        None,
                        None,
                        allowed_worker_ids,
                        routing_constraints,
                    )
                    .await?;
                match outcome {
                    crate::kv_router::FindBestMatchOutcome::Routed { worker, .. } => {
                        Ok(PrefillQueryOutcome::Routed {
                            worker_id: worker.worker_id,
                            dp_rank: Some(worker.dp_rank),
                        })
                    }
                    crate::kv_router::FindBestMatchOutcome::QueueRejected { rejection } => {
                        Ok(PrefillQueryOutcome::QueueRejected { rejection })
                    }
                }
            }
            InnerPrefillRouter::SimpleRouter(router) => {
                let worker_id = router
                    .peek_next_worker()
                    .ok_or_else(|| anyhow::anyhow!("No workers available for prefill"))?;
                Ok(PrefillQueryOutcome::Routed {
                    worker_id,
                    dp_rank: None,
                })
            }
        }
    }

    pub fn register_workers(&self, worker_ids: &HashSet<WorkerId>) {
        if let Some(InnerPrefillRouter::KvRouter(router)) = self.prefill_router.get() {
            router.chooser.register_workers(worker_ids);
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, OnceLock, atomic::AtomicU8};

    use dynamo_runtime::{
        DistributedRuntime, Runtime,
        distributed::DistributedConfig,
        pipeline::{PushRouter, RouterMode},
    };
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{
        discovery::ModelManager,
        kv_router::metrics::PREFILL_ASSIGNMENT_METRICS,
        protocols::common::llm_backend::{LLMEngineOutput, PreprocessedRequest},
        session_affinity::SessionAffinityPushRouter,
    };

    #[tokio::test]
    async fn advisory_query_does_not_increment_assignment_counters() {
        let runtime = Runtime::from_current().unwrap();
        let distributed =
            DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
                .await
                .unwrap();
        let namespace = distributed
            .namespace(format!("prefill_query_metrics_{}", uuid::Uuid::new_v4()))
            .unwrap();
        let endpoint = namespace
            .component("test_component".to_string())
            .unwrap()
            .endpoint("test_endpoint".to_string());
        let client = endpoint.client().await.unwrap();
        endpoint.register_endpoint_instance().await.unwrap();
        let worker_id = client.wait_for_instances().await.unwrap()[0].id();
        let push_router = PushRouter::<
            PreprocessedRequest,
            dynamo_runtime::protocols::annotated::Annotated<LLMEngineOutput>,
        >::from_client(client, RouterMode::RoundRobin)
        .await
        .unwrap();
        let inner = InnerPrefillRouter::SimpleRouter(Arc::new(
            SessionAffinityPushRouter::new(push_router, None, false).unwrap(),
        ));
        let prefill_router = OnceLock::new();
        assert!(prefill_router.set(inner).is_ok());
        let router = PrefillRouter {
            prefill_router,
            model_manager: Arc::new(ModelManager::new()),
            endpoint_id: OnceLock::new(),
            cancel_token: CancellationToken::new(),
            router_mode: RouterMode::RoundRobin,
            enforce_disagg: true,
            session_affinity_ttl: None,
            prefill_load_estimator: None,
            model_name: "test".to_string(),
            namespace: "test".to_string(),
            is_eagle: false,
            lifecycle: AtomicU8::new(PrefillLifecycleState::Active as u8),
        };

        let worker_label = worker_id.to_string();
        let request_counter = PREFILL_ASSIGNMENT_METRICS
            .assigned_requests_total
            .with_label_values(&[worker_label.as_str(), "none"]);
        let token_counter = PREFILL_ASSIGNMENT_METRICS
            .assigned_input_tokens_total
            .with_label_values(&[worker_label.as_str(), "none"]);
        let before_requests = request_counter.get();
        let before_tokens = token_counter.get();

        let outcome = router
            .query_prefill_worker(
                &[1, 2, 3],
                None,
                None,
                0.0,
                0,
                None,
                RoutingConstraints::default(),
            )
            .await
            .unwrap();

        assert!(matches!(
            outcome,
            PrefillQueryOutcome::Routed {
                worker_id: selected,
                dp_rank: None,
            } if selected == worker_id
        ));
        assert_eq!(request_counter.get(), before_requests);
        assert_eq!(token_counter.get(), before_tokens);

        runtime.shutdown();
    }
}
