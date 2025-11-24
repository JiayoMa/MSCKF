# KalmanNet for MSCKF

This module implements KalmanNet, a neural network-based approach for computing the Kalman gain in the MSCKF (Multi-State Constraint Kalman Filter) framework.

## Overview

KalmanNet replaces the traditional Kalman gain computation:
```
K = P * H' / (H * P * H' + R)
```

with a learned neural network that can capture non-linear dynamics and model uncertainties that are difficult to specify analytically.

## Architecture

The KalmanNet implementation uses:
- **Two GRU (Gated Recurrent Unit) networks**:
  - GRU 1: Processes the innovation (measurement residual)
  - GRU 2: Processes the state estimate
- **Fully connected layers**: Combine GRU outputs to produce the Kalman gain matrix

## Files

- `KalmanNetNN.m`: Main KalmanNet class implementation
- `computeKalmanGainKalmanNet.m`: Interface function for computing Kalman gain using KalmanNet
- `kalmannet_msckf_model.mat`: Pre-trained model weights (generated after training)

## Usage

### Running MSCKF with KalmanNet

Simply run the modified MSCKF script:
```matlab
cd msckf
MSCKF_KalmanNet
```

### Training KalmanNet

To train KalmanNet on your dataset:
```matlab
cd msckf
trainKalmanNet
```

### Configuration

In `MSCKF_KalmanNet.m`, you can adjust:
- `msckfParams.useKalmanNet`: Set to `true` to use KalmanNet, `false` for traditional EKF
- `msckfParams.kalmanNetBlend`: Blend factor (0.0 = traditional EKF, 1.0 = full KalmanNet)

## Reference

KalmanNet is based on:
- "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics" by Revach et al.

## Notes

- The neural network is initialized with Xavier initialization
- Hidden states are reset at the beginning of each sequence
- The model can be saved and loaded using `saveModel()` and `loadModel()` methods
