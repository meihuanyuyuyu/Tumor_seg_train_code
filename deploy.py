import torch
import model

dummy_in = torch.rand(1,3,512,512,dtype=torch.float32)
net = model.Unet_pp_width(2)
net.load_state_dict(torch.load('parameters/unet_pp_width_ds_0.2lamd.pt'))
torch.onnx.export(net,dummy_in,'test_model.onnx')