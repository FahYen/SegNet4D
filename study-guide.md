# Study Guide: CNN → Sparse 3D Conv → FPGA Acceleration

A practical reading order for catching up on the algorithmic and hardware background
needed to accelerate SegNet4D (or a subgraph of it) on an FPGA via HLS.

---

## 1. CNN Refresher (1–2 hours)

Core concepts: convolution, pooling, stride, channels, fully-connected layers.

- [Stanford CS231n — Convolutional Networks](https://cs231n.github.io/convolutional-networks/)
Read the convolution, pooling, and FC sections. Skip the training/backprop parts
if you only care about inference.

## 2. Sparse 3D Convolution (1–2 hours)

Why LiDAR point clouds need sparse representations, and how `spconv` works
(rulebook, gather, GEMM, scatter).

- [Submanifold Sparse Convolutional Networks — Graham & van der Maaten, 2017](https://arxiv.org/abs/1706.01307)
Sections 1–3 and the figures are the key parts.
- [spconv GitHub repository](https://github.com/traveller59/spconv)
The README explains the API that SegNet4D calls throughout
`models/backbones_3d/instance_aware_backbone.py`.

## 3. BEV-Based LiDAR Detection (1 hour)

These architectures flatten the 3D voxel grid into a 2D bird's-eye-view image
and run a standard dense 2D CNN — much easier to map onto an FPGA.

- [PointPillars — Lang et al., 2019](https://arxiv.org/abs/1812.05784)
Encodes LiDAR into vertical pillars, then a 2D backbone.
- [CenterPoint — Yin et al., 2021](https://arxiv.org/abs/2006.11275)
Center-based 3D object detection built on a BEV feature map.

## 4. HLS for CNN Accelerators (2–4 hours)

The bridge between "I have a trained CNN" and "it runs on FPGA fabric."
Covers loop tiling, pipelining, fixed-point quantization, and AXI interfaces.

- [Parallel Programming for FPGAs — Kastner et al. (free online book)](https://kastner.ucsd.edu/hlsbook/)
Chapters 1–4 cover the essentials. Chapter 5 (convolution case study) is
directly applicable.
- [Vitis HLS User Guide (AMD/Xilinx)](https://docs.amd.com/r/en-US/ug1399-vitis-hls)
Reference for pragmas (`#pragma HLS PIPELINE`, `UNROLL`, `ARRAY_PARTITION`, etc.).

## 5. Quantization for Deployment (1 hour)

Trained weights are FP32; FPGAs prefer INT8/INT16 for throughput and resource use.

- [PyTorch Quantization Overview](https://pytorch.org/docs/stable/quantization.html)
Post-training quantization and quantization-aware training.
- [Brevitas (Xilinx/AMD quantization library)](https://github.com/Xilinx/brevitas)
Purpose-built for quantizing models destined for FPGA deployment.

## 6. SegNet4D Codebase Walkthrough

Trace a single forward pass through the repository:

1. **Entry point:** `models/models.py` — `SegNet4D` (Lightning module) and `SegNet4D_Model` (nn.Module).
2. **Voxelization:** `models/backbones_3d/voxel_generate.py` — raw points → sparse voxel grid.
3. **Voxel feature encoding:** `models/backbones_2d/mean_vfe.py` — per-voxel mean pooling.
4. **3D sparse backbone:** `models/backbones_3d/instance_aware_backbone.py` — encoder (sparse conv + attention) and decoder (transposed sparse conv).
5. **BEV compression:** `models/backbones_2d/height_compression.py` — squash 3D features to 2D.
6. **2D BEV backbone:** `models/backbones_2d/base_bev_backbone.py` — dense 2D conv stack (FPGA-friendly candidate).
7. **Detection head:** `models/backbones_2d/center_head.py` — bounding-box regression.
8. **Segmentation heads:** MOS + semantic heads inside `instance_aware_backbone.py` (lines ~493–501).
9. **Fusion:** `models/backbones_3d/MSFM.py` — MOS–semantic feature fusion.
10. **Losses:** `models/loss.py` — `MOSLoss`, `SemanticLoss`.
11. **Config:** `config/semantickitti/semantickitti_config.yaml` — all hyperparameters.

## 7. Supplementary / Optional

- [FINN (Xilinx fast neural network inference)](https://github.com/Xilinx/finn)
End-to-end framework for compiling quantized binary/low-bit NNs to FPGA.
- [hls4ml](https://github.com/fastmachinelearning/hls4ml)
Converts Keras/PyTorch models to HLS for small networks (mostly MLPs and small CNNs).
- [VTA: Versatile Tensor Accelerator (Apache TVM)](https://tvm.apache.org/docs/topic/vta/index.html)
Open-source deep learning accelerator stack targeting FPGAs.
