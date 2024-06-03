# Multiscale geometric window transformer for orthodontic teeth point cloud registration
![Image text](Pipline.png)

### Introduction
  Digital orthodontic treatment monitoring has been gaining increasing attention in the past decade. However, current methods based on deep learning still face difficult challenges. Transformer, due to its excellent ability to model long-term dependencies, can be applied to the task of tooth point cloud registration. Nonetheless, most transformer-based point cloud registration networks suffer from two problems. First, they lack the embedding of credible geometric information, resulting in learned features that are not geometrically discriminative and blur the boundary between inliers and outliers. Second, the attention mechanism lacks continuous downsampling during geometric transformation invariant feature extraction at the superpixel level, thereby limiting the field of view and potentially limiting the model's perception of local and global information. In this paper, we propose GeoSwin, which uses a novel geometric window transformer to achieve accurate registration of tooth point clouds in different stages of orthodontic treatment. This method uses the point distance, normal vector angle, and bidirectional spatial angular distances as the input geometric embedding of transformer, and then uses a proposed variable multiscale attention mechanism to achieve geometric information perception from local to global perspectives. Experiments on the Shing3D Dental Dataset demonstrate the effectiveness of our approach and that it outperforms other state-of-the-art approaches across multiple metrics.

### News
Paper is available at [it](https://link.springer.com/article/10.1007/s00530-024-01369-x).

### Installation
Our code is largely based on [Geotransformer](https://github.com/qinzheng93/GeoTransformer), and the environment installation process can be referred to as in [it](https://github.com/qinzheng93/GeoTransformer). Here, we express our gratitude for their outstanding work.

### Code
Please replace file ../geotransformer/modules/geotransformer/geotransformer.py with geotransformer.py in the Git repository.

Please replace file ../geotransformer/modules/transformer/rpe_transformer.py with rpe_transformer.py in the Git repository.

Please replace file ../geotransformer/modules/geotransformer/conditional_transformer.py with conditional_transformer.py in the Git repository.

### Pre-trained Weights 
We provide pre-trained weights in the release page.

### Testing in Shining3D and Oral3 Dental Datasets 
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/Shining3D.pth.tar
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/Oral3.pth.tar
```
### Testing in Kitti odometry 
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --snapshot=../../weights/Kitti.pth.tar
```
## Citation
```
Wang, H., Tian, Y., Xu, Y. et al. Multiscale geometric window transformer for orthodontic teeth point cloud registration. Multimedia Systems 30, 164 (2024). https://doi.org/10.1007/s00530-024-01369-x
```

## Acknowledgements
- [Geotransformer](https://github.com/qinzheng93/GeoTransformer)








