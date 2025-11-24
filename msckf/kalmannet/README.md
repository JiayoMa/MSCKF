# KalmanNet for MSCKF

This module implements KalmanNet, a neural network-based approach for computing the Kalman gain in the MSCKF (Multi-State Constraint Kalman Filter) framework.

## Overview

KalmanNet replaces the traditional Kalman gain computation:
```matlab
K = P * H' / (H * P * H' + R)
```

with a learned neural network that can capture non-linear dynamics and model uncertainties that are difficult to specify analytically.

## Architecture

The KalmanNet implementation uses:
- **Two GRU (Gated Recurrent Unit) networks**:
  - GRU 1: Processes the innovation (measurement residual)
  - GRU 2: Processes the state estimate
- **Fully connected layers**: Combine GRU outputs to produce the Kalman gain matrix

```
Innovation (r_n) ──► GRU 1 ──┐
                             ├──► FC1 ──► FC2 ──► Kalman Gain (K)
State Estimate ────► GRU 2 ──┘
```

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

The training script will:
1. Collect training data by running the traditional EKF-based MSCKF
2. Use the traditional Kalman gains as supervision targets
3. Train the KalmanNet to approximate the optimal Kalman gain
4. Save the trained model to `kalmannet/kalmannet_msckf_model.mat`

### Configuration

In `MSCKF_KalmanNet.m`, you can adjust:
- `msckfParams.useKalmanNet`: Set to `true` to use KalmanNet, `false` for traditional EKF
- `msckfParams.kalmanNetBlend`: Blend factor (0.0 = traditional EKF, 1.0 = full KalmanNet)
- `hiddenDim`: Hidden layer dimension of the GRU networks (default: 64)
- `baseStateDim`: Expected state dimension (default: 72 = 12 IMU + 6*10 camera states)
- `baseObsDim`: Expected observation dimension (default: 20)

## API

### KalmanNetNN Class

```matlab
% Create a new KalmanNet instance
kalmanNet = KalmanNetNN(stateDim, obsDim, hiddenDim);

% Compute Kalman gain
K = kalmanNet.computeKalmanGain(innovation, stateEstimate, H);

% Save trained model
kalmanNet.saveModel('model.mat');

% Load pre-trained model
kalmanNet.loadModel('model.mat');

% Reset hidden states (call at start of new sequence)
kalmanNet.resetHiddenStates();
```

## Reference

KalmanNet is based on:
- "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics" by Revach et al., IEEE Transactions on Signal Processing, 2022.

## Notes

- The neural network is initialized with Xavier initialization
- Hidden states are reset at the beginning of each sequence
- The model can be saved and loaded using `saveModel()` and `loadModel()` methods
- Fallback to traditional EKF is automatic when dimension mismatches occur
- For production use, consider using MATLAB's Deep Learning Toolbox for GPU acceleration
