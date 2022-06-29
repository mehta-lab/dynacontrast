import os
import cellpose.models as models
model_types = ['nuclei', 'cyto', 'cyto2']
model_list = os.listdir(models.model_dir)
device, gpu = models.assign_device(True, False)
gpu=False
for model_type in model_types:
    model = models.Cellpose(gpu=gpu, device=device, model_type=model_type,
                        torch=True)
    for model_path in model.cp.pretrained_model:
        model.cp.net.load_model(model_path, cpu=True)
        model.cp.net.save_model(model_path, use_zip=False)
