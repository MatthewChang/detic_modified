python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input desk.jpg --output out.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-10-49/*.jpg --output vis/output --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

mkdir vis/output/___; python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/___/*.jpg --output vis/output/___ --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth


mkdir vis/output/2023-07-05-16-10-49; python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-10-49/*.jpg --output vis/output/2023-07-05-16-10-49 --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
mkdir vis/output/2023-07-05-16-17-28; python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-17-28/*.jpg --output vis/output/2023-07-05-16-17-28 --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
mkdir vis/output/2023-07-05-16-23-13; python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-23-13/*.jpg --output vis/output/2023-07-05-16-23-13 --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
mkdir vis/output/2023-07-05-16-29-54; python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-29-54/*.jpg --output vis/output/2023-07-05-16-29-54 --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

mkdir vis/output/___


mkdir vis/output/2023-07-05-16-10-49
mkdir vis/output/2023-07-05-16-17-28
mkdir vis/output/2023-07-05-16-23-13
mkdir vis/output/2023-07-05-16-29-54

python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-10-49/*.jpg --output vis/output/2023-07-05-16-10-49 --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

python extract_slices.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-29-54/*.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth

python extract_slices.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/___/*.jpg --vocabulary lvis --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth



test ___ 
test ___ 
test ___ 


python extract_slices.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-10-49/*.jpg --vocabulary lvis --obj-class cup --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth 
python extract_slices.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-17-28/*.jpg --vocabulary lvis --obj-class cup --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
python extract_slices.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-23-13/*.jpg --vocabulary lvis --obj-class cup --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth
python extract_slices.py --config-file configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml --input ./vis/frames/2023-07-05-16-29-54/*.jpg --vocabulary lvis --obj-class cup --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth


mv *.png ./grey

python reid/benchmark.py --method clip --pre-process aspect_resize --mask
0.526
python reid/benchmark.py --method clip --pre-process aspect_resize
0.448
python reid/benchmark.py --method clip --pre-process paste_center --mask
0.607
python reid/benchmark.py --method clip --pre-process paste_center
0.486
python reid/benchmark.py --method clip --mask
0.550
python reid/benchmark.py --method clip 
0.442

python reid/benchmark.py --method superglue --pre-process aspect_resize --mask
0.494
python reid/benchmark.py --method superglue --pre-process aspect_resize
0.499
python reid/benchmark.py --method superglue --pre-process paste_center --mask
0.231
python reid/benchmark.py --method superglue --pre-process paste_center
0.201
python reid/benchmark.py --method superglue --mask
0.180
python reid/benchmark.py --method superglue 
0.195

baseline 0.146

python reid/benchmark.py --method clip --pre-process paste_center --mask --vis
