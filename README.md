# AutoSlim: Towards One-Shot Architecture Search for Channel Numbers with TF2
- Implementation of Autoslim using Tensorflow2
- At now, source code is almost verified, but it still has redundancy and low readability.
- At now, I'm working with Mobilenet-v2, so other network architectures are not implemented fully.

## Note that
- This repositories hyper-parameter setting is different from the authors' one due to lack of explanation or limitation of the hardware.
- The authors have not published their greedily slimming codes, so I implement it naively. It is probably different from the authors,' but it works due to the power of slimmable network.

## Paper's abstract
We study how to set channel numbers in a neural network to achieve better accuracy under constrained resources (e.g., FLOPs, latency, memory footprint or model size). A simple and one-shot solution, named AutoSlim, is presented. Instead of training many network samples and searching with reinforcement learning, we train a single slimmable network to approximate the network accuracy of different channel configurations. We then iteratively evaluate the trained slimmable model and greedily slim the layer with minimal accuracy drop. By this single pass, we can obtain the optimized channel configurations under different resource constraints. We present experiments with MobileNet v1, MobileNet v2, ResNet-50 and RL-searched MNasNet on ImageNet classification. We show significant improvements over their default channel configurations. We also achieve better accuracy than recent channel pruning methods and neural architecture search methods. Notably, by setting optimized channel numbers, our AutoSlim-MobileNet-v2 at 305M FLOPs achieves 74.2% top-1 accuracy, 2.4% better than default MobileNet-v2 (301M FLOPs), and even 0.2% better than RL-searched MNasNet (317M FLOPs). Our AutoSlim-ResNet-50 at 570M FLOPs, without depthwise convolutions, achieves 1.3% better accuracy than MobileNet-v1 (569M FLOPs).

## Requirements
- Tensorflow > 2.0
- Scipy

## Run
- Run train_w_slimming.py with slimmable = True

## Experimental results
- I only use CIFAR10 dataset due to my low hardware performance.
- All the training configuration is probably not optimal.

### MobileNet-v2 (will be uploaded soon)
- Table and Plots...
- Comparing original and slimmed networks' layer depths via Box plot.

## Reference
```
@article{yu2019autoslim,
  title={AutoSlim: Towards One-Shot Architecture Search for Channel Numbers},
  author={Yu, Jiahui and Huang, Thomas},
  journal={arXiv preprint arXiv:1903.11728},
  volume={8},
  year={2019}
}
```

## Original project page
https://github.com/JiahuiYu/slimmable_networks
