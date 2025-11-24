function K = computeKalmanGainKalmanNet(kalmanNet, T_H, P, R_n, r_n, stateDim)
% computeKalmanGainKalmanNet: Compute Kalman gain using KalmanNet
%
% This function replaces the traditional Kalman gain computation:
%   K = (P*T_H') / (T_H*P*T_H' + R_n)
% with a neural network-based approach using KalmanNet
%
% Inputs:
%   kalmanNet: KalmanNetNN object (neural network)
%   T_H: Observation Jacobian matrix (after QR decomposition)
%   P: State covariance matrix
%   R_n: Measurement noise covariance matrix
%   r_n: Innovation (measurement residual)
%   stateDim: Total state dimension
%
% Output:
%   K: Kalman gain matrix (size: [size(P,1), size(T_H,1)])

    % Extract state estimate from covariance (use diagonal as proxy for state uncertainty)
    stateEstimate = sqrt(diag(P));
    
    % Use innovation as the primary input to KalmanNet
    innovation = r_n;
    
    % Compute Kalman gain using neural network
    K = kalmanNet.computeKalmanGain(innovation, stateEstimate, T_H);
    
    % Expected dimensions: K should be [size(P,1), size(T_H,1)]
    % This matches the traditional formula K = P*T_H' / (T_H*P*T_H' + R_n)
    rows_expected = size(P, 1);
    cols_expected = size(T_H, 1);
    [rows_K, cols_K] = size(K);
    
    if rows_K ~= rows_expected || cols_K ~= cols_expected
        % Fall back to traditional computation if dimensions don't match
        % This ensures robustness during the transition
        K_traditional = (P * T_H') / (T_H * P * T_H' + R_n);
        
        % Blend KalmanNet output with traditional if sizes match partially
        if rows_K <= rows_expected && cols_K <= cols_expected
            K_full = K_traditional;
            K_full(1:rows_K, 1:cols_K) = K;
            K = K_full;
        else
            K = K_traditional;
        end
    end
end
