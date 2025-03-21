# NFD_toy

This is a toy model that follows the workflow described in the paper "3D Neural Field Generation using Triplane Diffusion" by Shue et al. The implementation provides a simplified yet functional version of the Neural Field Diffusion framework.

## Workflow

### 1. Setup
- Install dependencies: PyTorch, trimesh, PyMCubes, matplotlib, numpy

```
pip install torch numpy matplotlib trimesh PyMCubes tqdm
```

### 2. Encoder & Decoder
The model follows a triplane representation approach where 3D information is decomposed into three orthogonal feature planes.

#### Encoder
- **Network**: Three 2D feature planes with a shared MLP decoder
- **Input**: 3D shape represented as meshes (.off files)
- **Output**: Three feature planes (XY, YZ, XZ) that encode 3D information
- **Loss**: Binary Cross Entropy with Logits (BCEWithLogitsLoss) for occupancy prediction
- **Parameters**:
  - Feature dimensions: 32
  - Resolution: 128x128
  - Batch size: 10000
  - Learning rate: 1e-4
  - Epochs: 200

#### Decoder (MiniTriplane)
- **Network**: FourierFeatureTransform + MLP
- **Function**: Converts triplane features back to 3D occupancy fields
- **Parameters**:
  - Marching Cubes resolution: 128
  - Occupancy threshold: 0.0

### 3. Diffusion Model
The model uses denoising diffusion probabilistic modeling (DDPM) to learn and generate triplane features.

- **Backbone**: UNet with time embedding and self-attention
- **Architecture**:
    Downsampling path: ResBlocks + Downsampling
    Bottleneck: ResBlocks + Self-Attention
    Upsampling path: ResBlocks + Upsampling + Skip connections
- **Diffusion Process**: Gaussian diffusion with 1000 timesteps

### 4. Dataset
- ModelNet40 dataset (.off files)

### 4. Experiment Workflow
1. Load 3D models from ModelNet40
2. Sample points from meshes and compute occupancy values
3. Train encoder to convert 3D shapes to triplane features
4. Create TriplaneDataset by stacking features in channel dimension
5. Train diffusion model on triplane features dataset
6. Generate new triplane features using the diffusion model
7. Convert generated features to 3D models using the decoder
8. Save and visualize results

## Results
Results are saved in the `generated_features` directory:
- Triplane visualizations (PNG images)
- 3D models (OBJ files)
- Feature data (NPY files)

### output
results/triplane_features/ - Training features  
results/triplane_features/reconstructions/ - Reconstructed meshes from training features  
results/generated_samples/ - Generated models  
checkpoints/ - Saved model checkpoints  

## Further Work
This toy model does not include the regularization techniques required for a smooth and perfect model. The implementation focuses on demonstrating the core concepts rather than achieving state-of-the-art results.

Future directions:
- Add regularization terms to improve surface quality
- Implement conditioning for controlled generation
- Compare with other 3D generation workflows that explore latent space (e.g., VAEs)
- Train on larger datasets like ShapeNet for more diverse generation
- Implement evaluation metrics (FID, Chamfer distance, etc.) to benchmark quality
- Optimize the implementation for faster training and inference


## Code stture:

## Citation
If you use this code for your research, please cite the original paper:
```
@inproceedings{shue2023nfd,
  title={3D Neural Field Generation using Triplane Diffusion},
  author={Shue, J. Ryan and Chan, Eric Ryan and Po, Ryan and Ankner, Zachary and Wu, Jiajun and Wetzstein, Gordon},
  booktitle={arxiv},
  year={2023}
}
```

## License
[MIT License](LICENSE)