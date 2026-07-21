use ek_base::error::EKResult;
use expert_ort::OnnxFFN;
use expert_torch::TorchFFN;
use meta::{Expert, ExpertWeight};
use safetensors::tensor::TensorView;
use tracing::instrument;

use crate::{
    backend::{EkTensor, ggml::GgmlTensor, torch::TchTensor},
    ffn::expert_ggml::GgmlFFN,
    worker::profile::NvtxRange,
    x::{self},
};

pub mod expert_ggml;
#[allow(dead_code)]
pub mod expert_ort;
pub mod expert_torch;
pub mod meta;

pub enum ExpertBackend {
    Torch(TorchFFN),
    OnnxF32(OnnxFFN<f32>),
    Ggml(GgmlFFN),
}

impl ExpertBackend {
    pub async fn build<'a>(
        instance: x::EKInstance,
        tensor: &'a safetensors::SafeTensors<'a>,
    ) -> EKResult<ExpertBackend> {
        let backend = match instance.backend {
            x::ExpertBackendType::Torch => {
                let weight = ExpertWeight::<TchTensor>::from_safetensor(tensor, instance.device)?;
                ExpertBackend::Torch(TorchFFN::construct(instance, weight)?)
            }
            x::ExpertBackendType::Ggml => {
                let weight: ExpertWeight<GgmlTensor> =
                    ExpertWeight::<GgmlTensor>::from_safetensor(tensor, instance.device)?;
                ExpertBackend::Ggml(expert_ggml::GgmlFFN::construct(instance, weight)?)
            }
            x::ExpertBackendType::Onnx => todo!(),
        };
        Ok(backend)
    }
}

impl ExpertBackend {
    #[instrument(skip(self, view))]
    pub fn forward(&self, view: &TensorView) -> EKResult<Vec<u8>> {
        match self {
            ExpertBackend::Torch(exp) => {
                let inp = {
                    let _range = NvtxRange::new("EKR:WORKER_INPUT_TENSOR");
                    TchTensor::from_tensor_view(view)
                };
                let shape = inp.inner().size();
                let inp = {
                    let _range = NvtxRange::new("EKR:WORKER_INPUT_DEVICE");
                    inp.to_device(exp.device())
                };
                log::debug!("input shape {shape:?}");
                assert!(shape.len() == 2);
                let output = {
                    let _range = NvtxRange::new("EKR:WORKER_EXPERT_MATH");
                    exp.forward(&inp)
                };
                let _range = NvtxRange::new("EKR:WORKER_OUTPUT_ST_SAVE");
                Ok(output.serialize())
            }
            ExpertBackend::OnnxF32(_exp) => {
                todo!()
            }
            ExpertBackend::Ggml(exp) => {
                let inp = {
                    let _range = NvtxRange::new("EKR:WORKER_INPUT_TENSOR");
                    GgmlTensor::from_tensor_view(view)
                };
                let shape = inp.shape();
                log::debug!("input shape {shape:?}");
                assert!(shape.len() == 2);
                let output = {
                    let _range = NvtxRange::new("EKR:WORKER_EXPERT_MATH");
                    exp.forward(&inp)
                };
                let _range = NvtxRange::new("EKR:WORKER_OUTPUT_ST_SAVE");
                Ok(output.serialize())
            }
        }
    }
}
