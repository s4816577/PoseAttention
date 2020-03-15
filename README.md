# PoseAttention implementation in PyTorch
This is the PyTorch implementation for PoseAttention, developed based on PoseLSTM(https://github.com/hazirbas/poselstm-pytorch) code.

## Prerequisites
- Linux
- Python 3.6.9
- CPU or NVIDIA GPU + CUDA CuDNN
- PyTorch 1.4.0

### PoseAttention train/test
- Download a Cambridge Landscape dataset (e.g. [KingsCollege](http://mi.eng.cam.ac.uk/projects/relocalisation/#dataset)) under datasets/ folder.

- Train a model:
```bash
python train.py --model poselstm --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/attn_lw1018 --gpu 0 --learn_weight --niter 2000 --save_epoch_freq 1 
```

- Test the model:
```bash
python test.py --model poselstm  --dataroot ./datasets/KingsCollege --name posenet/KingsCollege/attn_lw1018 --gpu 0
```
The test errors will be saved to a text file under `./results/posenet/KingsCollege/attn_lw1018/`.

### Initialize the network with the pretrained googlenet trained on the Places dataset
If you would like to initialize the network with the pretrained weights, download the places-googlenet.pickle file under the *pretrained_models/* folder:
``` bash
wget https://vision.in.tum.de/webarchive/hazirbas/poselstm-pytorch/places-googlenet.pickle
```

## Citation
```
@inproceedings{PoseNet15,
  title={PoseNet: A Convolutional Network for Real-Time 6-DOF Camera Relocalization},
  author={Alex Kendall, Matthew Grimes and Roberto Cipolla },
  journal={ICCV},
  year={2015}
}
@inproceedings{PoseLSTM17,
  author = {Florian Walch and Caner Hazirbas and Laura Leal-Taixé and Torsten Sattler and Sebastian Hilsenbeck and Daniel Cremers},
  title = {Image-based localization using LSTMs for structured feature correlation},
  month = {October},
  year = {2017},
  booktitle = {ICCV},
  eprint = {1611.07890},
  url = {https://github.com/NavVisResearch/NavVis-Indoor-Dataset},
}