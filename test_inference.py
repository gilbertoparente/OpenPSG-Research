import mmcv
from mmcv import Config
from mmdet.apis import init_detector, inference_detector
from openpsg.utils.utils import show_result

cfg = Config.fromfile("configs/psgtr/psgtr_r50_psg_inference.py")

model = init_detector(
    cfg,
    "work_dirs/checkpoints/epoch_60.pth",
    device="cpu"
)

print("Modelo carregado com sucesso!")

img = "test.jpg"

print("A iniciar inferência...")

result = inference_detector(model, img)

print("Inferência concluída!")

show_result(
    img,
    result,
    is_one_stage=True,
    num_rel=5,
    out_dir="resultado",
    out_file="resultado/output.png"
)

print("Resultados guardados na pasta 'resultado'.")