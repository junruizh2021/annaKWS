import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision
import yaml

from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint

ckpt_path = 'model/hixiaowen/avg_30.pt'
config_path = 'model/hixiaowen/config.yaml'
onnx_model_path = 'model/hixiaowen/annaKWS.onnx'
tfpb_model_path = 'model/hixiaowen/annaKWS.pb'


def np_compare(golden, predicted):
    assert (golden.shape == predicted.shape), 'golden and predicted must have the same shape'
    return np.allclose(golden, predicted, rtol=1.e-2, atol=1.e-2)


print("load torch model")

with open(config_path, 'r') as fin:
    configs = yaml.load(fin, Loader=yaml.FullLoader)
model = init_model(configs['model'])
load_checkpoint(model, ckpt_path)
device = torch.device('cpu')
model = model.to(device)
model.eval()
dummy_input_feature = np.random.randn(1, 10, 400).astype(np.float32)
dummy_input_cache = np.random.randn(1, 128, 11, 4).astype(np.float32)

dummy_input_feature_torch_tensor = torch.tensor(dummy_input_feature).to(device)
dummy_input_cache_torch_tensor = torch.tensor(dummy_input_cache).to(device)

torch_logits, torch_cache = model(dummy_input_feature_torch_tensor, dummy_input_cache_torch_tensor)

# print(model)
input_names = ["input_feature", "input_cache"]
output_names = ["output", "output_cache"]
torch.onnx.export(model,
                  (dummy_input_feature_torch_tensor, dummy_input_cache_torch_tensor),
                  onnx_model_path,
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)

print('Export ONNX!')

onnx_model = onnx.load(onnx_model_path)
ort_session = ort.InferenceSession(onnx_model_path)
onnx_logits, onnx_cache = ort_session.run(output_names, {'input_feature': dummy_input_feature, 'input_cache': dummy_input_cache})

print("compare torch_logits and onnx_logits = {}".format(np_compare(torch_logits.detach().numpy(), onnx_logits)))
print("compare torch_cache and onnx_cache = {}".format(np_compare(torch_cache.detach().numpy(), onnx_cache)))