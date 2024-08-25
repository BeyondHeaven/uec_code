# Unsupervised Exposure Correction (UEC) Documentation

## Prerequisite

### Dataset
To access the Afifi's Exposure-errors Dataset, please follow the instructions at [this link](https://github.com/mahmoudnafifi/Exposure_Correction#dataset).

For our Radiometry Correction Dataset, please check [this link](https://pan.baidu.com/s/1iQgFoLWZXW7eswaM5U3_6Q?pwd=z1yh)

### Package Installation
Install the necessary Python packages by running the following command:
```shell
pip install -r requirements.txt
```
### Pretrained Models
In the "checkpoints" directory, we have made available two ".pth" files, representing the outcomes of training on both Afifi's Exposure-errors Dataset and our Radiometry Correction Dataset, respectively.


## Training

Make sure to replace `dataset_root` with the actual path where you downloaded the dataset.

```shell
python train.py  --name exposure --model uec --dataset_mode exposure --load_size 448 --preprocess resize_and_crop --gpu_ids 2  --save_epoch_freq 1 --lr 1e-4 --beta1 0.9 --lr_policy step --lr_decay_iters 6574200 --dataset_root ../data/exposure_dataset/INPUT_IMAGES/
```
If you are using the Radiometry Correction Dataset, set `--dataset_mode` to `fivek`.
We removed TVLoss because we found the performance to be better without it. The PSNR result is as following:

| EV    | -2     | -1     | 0      | +1     | +2     | +3     |
|-------|--------|--------|--------|--------|--------|--------|
| w/ TVLoss  | 22.577 | 20.528 | 18.336 | 17.820 | 15.752 | 15.138 |
| w/o TVLoss | 25.343 | 23.637 | 20.552 | 18.391 | 15.327 | 13.175 |

To reproduce the results from the paper, please run:
```shell
git reset --hard 9578ef19c250b349d2a247913af8e5e902e7f707
```



## Testing
Replace `dataset_root` with the actual path for testing:
```shell
python test.py  --name exposure-errors --model uec --dataset_mode fivektest --preprocess resize --load_size 256 --gpu_ids 2  --dataset_root ../data//exposure_dataset/test/ --ref_image_paths ../data/exposure_dataset/GT_IMAGES/a0001-jmac_DSC1459.jpg
```

The option `--ref_image_paths` is for choosing one reference for the final calibration.

## Evaluation

We perform evaluation using [PyIQA](https://github.com/chaofengc/IQA-PyTorch).
```shell
python inference_iqa_filelist.py -f ./results/file.txt -i ./results/exposure/test_latest/images/ -r <path-to-your-reference-images>
```
`file.txt` contains the mapping between input images and ground truth, , separated by tabs. If you need to resize the images, run the following command:

```shell
python script_name.py --img_size 256 --input_folder ./input_images --output_folder ./output_images
```
## Acknowledgement

We brrow some code from [BargainNet](https://github.com/bcmi/BargainNet), [NeurOP](https://github.com/amberwangyili/neurop) and [Pix2PixHD](https://github.com/NVIDIA/pix2pixHD). We would like to thanks the authors for sharing their excellent works.

## License Agreement

- This repository with the provided code and pretrained model is available for non-commercial research purposes only.