use std::{fmt::Debug, sync::Arc};

use ek_base::{config::get_ek_settings, utils::Defers};
use tokio::sync::{Mutex, mpsc};
use tracing::Instrument;

use crate::{
    controller::{
        executor::{Executor, get_executor},
        metrics::METRIC_CONTROLLER_LAYER,
    },
    proto::ek::worker::v1::{self, computation_service_server::ComputationService},
    worker::profile::NvtxAsyncRange,
};

pub struct ComputationProxyServiceImpl {
    executor: Arc<Mutex<dyn Executor + Send>>,
}
impl Debug for ComputationProxyServiceImpl {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationProxyServiceImpl").finish()
    }
}

#[async_trait::async_trait]
impl ComputationService for ComputationProxyServiceImpl {
    async fn forward(
        &self,
        request: tonic::Request<v1::ForwardReq>,
    ) -> Result<tonic::Response<v1::ForwardResp>, tonic::Status> {
        self.inner_controller_forward(request).await
    }
}

impl ComputationProxyServiceImpl {
    #[tracing::instrument(
        skip(self, request),
        level = "info",
        fields(method = "ComputationProxyServiceImpl::forward",)
    )]
    async fn inner_controller_forward(
        &self,
        request: tonic::Request<v1::ForwardReq>,
    ) -> Result<tonic::Response<v1::ForwardResp>, tonic::Status> {
        let seq_len = request.get_ref().sequences.len();
        let _request_range = NvtxAsyncRange::phase(
            "CONTROLLER_REQUEST",
            &request.get_ref().request_id,
            request.get_ref().microbatch_id,
            request.get_ref().layer_id,
        );
        log::info!(
            seq_len,
            request_id = request.get_ref().request_id.as_str(),
            microbatch_id = request.get_ref().microbatch_id,
            layer_id = request.get_ref().layer_id;
            "forward request in controller start"
        );
        let start = std::time::Instant::now();
        let settings = get_ek_settings();

        let cloned_start = start;
        let _d = Defers::defer(Box::new(move || {
            let elapsed = cloned_start.elapsed();
            // TODO: hardcode model name in metric
            METRIC_CONTROLLER_LAYER
                .with_label_values(&[settings.inference.model_name.as_str()])
                .observe(elapsed.as_micros() as f64);
        }));

        if request.get_ref().pipeline_enabled {
            let mut executor = crate::controller::executor::NaiveExecutor::new_pipeline();
            let mut rx = executor.submit(request.get_ref()).await?;
            executor.exec().await.map_err(|err| {
                log::error!("pipeline executor error: {err}");
                tonic::Status::internal(format!("pipeline executor error: {err}"))
            })?;
            return rx
                .recv()
                .await
                .map(|resp| tonic::Response::new(resp.as_ref().clone()))
                .ok_or_else(|| tonic::Status::internal("pipeline forward error: no data"));
        }

        let mut rx = {
            let mut lg = self.executor.lock().await;
            lg.submit(request.get_ref()).await?
        };

        let exec_bg = self.executor.clone();
        let (err_tx, mut err_rx) = mpsc::channel(1);

        tokio::spawn(
            async move {
                let mut lg = exec_bg.lock().await;
                let res = lg.exec().await;
                if let Err(err) = res {
                    log::error!("executor error: {err}");
                    err_tx.send(err).await.unwrap();
                }
            }
            .in_current_span(),
        );

        loop {
            tokio::select! {
                err = err_rx.recv() => {
                    if let Some(err) = err {
                        log::error!("executor error: {err:?}");
                        return Err(tonic::Status::internal(format!("executor error: {err:?}")))
                    }
                    continue
                }
                res = rx.recv() => {
                    let elapsed_ms=  start.elapsed().as_millis();
                    log::info!(elapsed_ms; "forward request in controller done" );
                    if let Some(resp) = res {
                        return Ok(tonic::Response::new(resp.as_ref().clone()));
                    } else {
                        return Err(tonic::Status::internal("forward error: no data"));
                    }
                }
            }
        }
    }
}

impl Default for ComputationProxyServiceImpl {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputationProxyServiceImpl {
    pub fn new() -> Self {
        Self {
            executor: get_executor(),
        }
    }
}
