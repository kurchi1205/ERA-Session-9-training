## Objective
To train a model on the CIFAR-10 data following the C1C2C3C4O architecture and achieve 85% accuracy within 200k parameters. RF must be more than 44.

## Approach
### Intuition
RF of 44 is a very high target. RF can be increased by dilated convolutions or strided convolutions. But strided convolution will affect jump parameter which will increase receptive field for the consecutive layers as well. So it is better to used strided convolution in the initial layers and dilated convolution towards the end. Depthwise separable convolutions are always followed by a 1x1 which do not increase the receptive field, so it is better to use it in the initial layers when the jump factor is low.

### Steps
- Data class is created for albumentation transforms (data.py)
- All the 3 transformations are applied on the training data
  - horizontal flip
  - shiftScaleRotate
  - coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
  - normalization
- test data just has normalization transform
- model is written with the above intuition in mind. (model.py). The first layer has depth wise separable convolution and dilated convolution as I have 2 more blocks to add strided convolution.  The layer before GAP also has dilated convolution. 
- training and test code are in utils_train.py
- model is training for 85 epochs.
- I have adam optimizer and cosine annealing scheduler to gradually decrease the learning rate.

## Results
- Model parameters: 191,653
- Receptive field: 57
- Validation accuracy: 85.62 %
