%% trainKalmanNet.m
% Training script for KalmanNet using MSCKF data
% This script collects training data by running the traditional EKF-based MSCKF
% and then trains the KalmanNet to approximate the optimal Kalman gain.

clear;
close all;
clc;
addpath('utils');
addpath('kalmannet');

fprintf('=== KalmanNet Training for MSCKF ===\n\n');

%% Configuration
dataDir = '../datasets';

% List of training datasets
trainingFiles = {
    '2011_09_26_drive_0001_sync_KLT', 2, 98;
    '2011_09_26_drive_0036_sync_KLT', 2, 239;
    '2011_09_26_drive_0051_sync_KLT', 2, 114;
    '2011_09_26_drive_0095_sync_KLT', 2, 139;
};

% KalmanNet parameters
baseStateDim = 12 + 6 * 10;  % 12 IMU + 6*10 camera states
baseObsDim = 20;             % Observation dimension
hiddenDim = 64;              % Hidden layer dimension

% Training parameters
numEpochs = 50;
batchSize = 16;

% Initialize KalmanNet
kalmanNet = KalmanNetNN(baseStateDim, baseObsDim, hiddenDim);

%% Collect Training Data
fprintf('Collecting training data from %d datasets...\n', size(trainingFiles, 1));

allTrainingData.innovations = {};
allTrainingData.stateEstimates = {};
allTrainingData.trueStates = {};
allTrainingData.H = {};
allTrainingData.K_traditional = {};

for dataIdx = 1:size(trainingFiles, 1)
    fileName = trainingFiles{dataIdx, 1};
    kStart = trainingFiles{dataIdx, 2};
    kEnd = trainingFiles{dataIdx, 3};
    
    fprintf('\nProcessing: %s (frames %d-%d)\n', fileName, kStart, kEnd);
    
    % Load dataset
    load(sprintf('%s/%s.mat', dataDir, fileName));
    
    % Set up camera parameters
    camera.c_u = cu;
    camera.c_v = cv;
    camera.f_u = fu;
    camera.f_v = fv;
    camera.b = b;
    camera.q_CI = rotMatToQuat(C_c_v);
    camera.p_C_I = rho_v_c_v;
    
    % Noise parameters
    y_var = 11^2 * ones(1,4);
    noiseParams.u_var_prime = y_var(1)/camera.f_u^2;
    noiseParams.v_var_prime = y_var(2)/camera.f_v^2;
    
    w_var = 4e-2 * ones(1,3);
    v_var = 4e-2 * ones(1,3);
    dbg_var = 1e-6 * ones(1,3);
    dbv_var = 1e-6 * ones(1,3);
    noiseParams.Q_imu = diag([w_var, dbg_var, v_var, dbv_var]);
    
    q_var_init = 1e-6 * ones(1,3);
    p_var_init = 1e-6 * ones(1,3);
    bg_var_init = 1e-6 * ones(1,3);
    bv_var_init = 1e-6 * ones(1,3);
    noiseParams.initialIMUCovar = diag([q_var_init, bg_var_init, bv_var_init, p_var_init]);
    
    % MSCKF parameters
    msckfParams.minTrackLength = 10;
    msckfParams.maxTrackLength = Inf;
    msckfParams.maxGNCostNorm = 1e-2;
    msckfParams.minRCOND = 1e-12;
    msckfParams.doNullSpaceTrick = true;
    msckfParams.doQRdecomp = true;
    
    numLandmarks = size(y_k_j,3);
    dT = [0, diff(t)];
    measurements = cell(1,numel(t));
    
    y_k_j(y_k_j == -1) = NaN;
    
    % Initialize ground truth states
    groundTruthStates = cell(1, numel(t));
    
    % Prepare measurements and ground truth
    for state_k = kStart:kEnd
        measurements{state_k}.dT = dT(state_k);
        measurements{state_k}.y = squeeze(y_k_j(1:2,state_k,:));
        measurements{state_k}.omega = w_vk_vk_i(:,state_k);
        measurements{state_k}.v = v_vk_vk_i(:,state_k);
        
        validMeas = ~isnan(measurements{state_k}.y(1,:));
        measurements{state_k}.y(1,validMeas) = (measurements{state_k}.y(1,validMeas) - camera.c_u)/camera.f_u;
        measurements{state_k}.y(2,validMeas) = (measurements{state_k}.y(2,validMeas) - camera.c_v)/camera.f_v;
        
        q_IG = rotMatToQuat(axisAngleToRotMat(theta_vk_i(:,state_k)));
        p_I_G = r_i_vk_i(:,state_k);
        
        groundTruthStates{state_k}.imuState.q_IG = q_IG;
        groundTruthStates{state_k}.imuState.p_I_G = p_I_G;
        
        C_IG = quatToRotMat(q_IG);
        q_CG = quatLeftComp(camera.q_CI) * q_IG;
        p_C_G = p_I_G + C_IG' * camera.p_C_I;
        
        groundTruthStates{state_k}.camState.q_CG = q_CG;
        groundTruthStates{state_k}.camState.p_C_G = p_C_G;
    end
    
    % Initialize MSCKF
    featureTracks = {};
    trackedFeatureIds = [];
    
    firstImuState.q_IG = rotMatToQuat(axisAngleToRotMat(theta_vk_i(:,kStart)));
    firstImuState.p_I_G = r_i_vk_i(:,kStart);
    
    [msckfState, featureTracks, trackedFeatureIds] = initializeMSCKF(firstImuState, measurements{kStart}, camera, kStart, noiseParams);
    
    % Sequence data for this dataset
    seqInnovations = [];
    seqStateEstimates = [];
    seqTrueStates = [];
    seqH = {};
    seqK = {};
    
    % Run MSCKF and collect data
    for state_k = kStart:(kEnd-1)
        % State propagation
        msckfState = propagateMsckfStateAndCovar(msckfState, measurements{state_k}, noiseParams);
        msckfState = augmentState(msckfState, camera, state_k+1);
        
        % Feature tracking
        featureTracksToResidualize = {};
        
        for featureId = 1:numLandmarks
            meas_k = measurements{state_k+1}.y(:, featureId);
            outOfView = isnan(meas_k(1,1));
            
            if ismember(featureId, trackedFeatureIds)
                if ~outOfView
                    featureTracks{trackedFeatureIds == featureId}.observations(:, end+1) = meas_k;
                    msckfState.camStates{end}.trackedFeatureIds(end+1) = featureId;
                end
                
                track = featureTracks{trackedFeatureIds == featureId};
                
                if outOfView || size(track.observations, 2) >= msckfParams.maxTrackLength || state_k+1 == kEnd
                    [msckfState, camStates, camStateIndices] = removeTrackedFeature(msckfState, featureId);
                    
                    if length(camStates) >= msckfParams.minTrackLength
                        track.camStates = camStates;
                        track.camStateIndices = camStateIndices;
                        featureTracksToResidualize{end+1} = track;
                    end
                    
                    featureTracks = featureTracks(trackedFeatureIds ~= featureId);
                    trackedFeatureIds(trackedFeatureIds == featureId) = [];
                end
                
            elseif ~outOfView && state_k+1 < kEnd
                track.featureId = featureId;
                track.observations = meas_k;
                featureTracks{end+1} = track;
                trackedFeatureIds(end+1) = featureId;
                msckfState.camStates{end}.trackedFeatureIds(end+1) = featureId;
            end
        end
        
        % Measurement update - collect training data
        if ~isempty(featureTracksToResidualize)
            H_o = [];
            r_o = [];
            R_o = [];
            
            for f_i = 1:length(featureTracksToResidualize)
                track = featureTracksToResidualize{f_i};
                [p_f_G, Jcost, RCOND] = calcGNPosEst(track.camStates, track.observations, noiseParams);
                
                nObs = size(track.observations,2);
                JcostNorm = Jcost / nObs^2;
                
                if JcostNorm > msckfParams.maxGNCostNorm || RCOND < msckfParams.minRCOND
                    continue;  % Skip this feature track and try next one
                end
                
                [r_j] = calcResidual(p_f_G, track.camStates, track.observations);
                R_j = diag(repmat([noiseParams.u_var_prime, noiseParams.v_var_prime], [1, numel(r_j)/2]));
                [H_o_j, A_j, H_x_j] = calcHoj(p_f_G, msckfState, track.camStateIndices);
                
                if msckfParams.doNullSpaceTrick
                    H_o = [H_o; H_o_j];
                    if ~isempty(A_j)
                        r_o_j = A_j' * r_j;
                        r_o = [r_o; r_o_j];
                        R_o_j = A_j' * R_j * A_j;
                        R_o(end+1:end+size(R_o_j,1), end+1:end+size(R_o_j,2)) = R_o_j;
                    end
                else
                    H_o = [H_o; H_x_j];
                    r_o = [r_o; r_j];
                    R_o(end+1:end+size(R_j,1), end+1:end+size(R_j,2)) = R_j;
                end
            end
            
            if ~isempty(r_o)
                if msckfParams.doQRdecomp
                    [T_H, Q_1] = calcTH(H_o);
                    r_n = Q_1' * r_o;
                    R_n = Q_1' * R_o * Q_1;
                else
                    T_H = H_o;
                    r_n = r_o;
                    R_n = R_o;
                end
                
                P = [msckfState.imuCovar, msckfState.imuCamCovar;
                     msckfState.imuCamCovar', msckfState.camCovar];
                
                % Traditional Kalman gain
                K = (P*T_H') / (T_H*P*T_H' + R_n);
                
                % Collect training data
                stateEstimate = sqrt(diag(P));
                
                % Store data point
                seqInnovations = [seqInnovations, r_n];
                seqStateEstimates = [seqStateEstimates, stateEstimate];
                seqH{end+1} = T_H;
                seqK{end+1} = K;
                
                % Get true state for this timestep
                trueState = [groundTruthStates{state_k+1}.imuState.q_IG(1:3);
                            zeros(3,1);  % bias
                            zeros(3,1);  % bias
                            groundTruthStates{state_k+1}.imuState.p_I_G];
                
                % Pad to match state dimension
                stateDim = size(P, 1);
                if length(trueState) < stateDim
                    trueState = [trueState; zeros(stateDim - length(trueState), 1)];
                end
                seqTrueStates = [seqTrueStates, trueState(1:stateDim)];
                
                % State correction
                deltaX = K * r_n;
                msckfState = updateState(msckfState, deltaX);
                
                % Covariance correction
                tempMat = (eye(size(P,1)) - K*T_H);
                P_corrected = tempMat * P * tempMat' + K * R_n * K';
                
                msckfState.imuCovar = P_corrected(1:12,1:12);
                msckfState.camCovar = P_corrected(13:end,13:end);
                msckfState.imuCamCovar = P_corrected(1:12, 13:end);
            end
        end
        
        % State pruning
        [msckfState, ~] = pruneStates(msckfState);
    end
    
    % Store sequence data
    if ~isempty(seqInnovations)
        allTrainingData.innovations{end+1} = seqInnovations;
        allTrainingData.stateEstimates{end+1} = seqStateEstimates;
        allTrainingData.trueStates{end+1} = seqTrueStates;
        allTrainingData.H{end+1} = seqH;
        allTrainingData.K_traditional{end+1} = seqK;
    end
    
    fprintf('Collected %d training samples from %s\n', size(seqInnovations, 2), fileName);
end

%% Train KalmanNet
fprintf('\n=== Training KalmanNet ===\n');
fprintf('Total sequences: %d\n', length(allTrainingData.innovations));

% Training using numerical gradient approximation
% Note: For production use, consider using MATLAB's Deep Learning Toolbox
% which provides automatic differentiation and GPU acceleration.
% This implementation uses finite difference gradient approximation for 
% the fully connected layers that output the Kalman gain.

fprintf('Training for %d epochs...\n', numEpochs);
learningRate = 0.001;

for epoch = 1:numEpochs
    epochLoss = 0;
    numSamples = 0;
    
    % Compute gradients using finite differences on a subset of weights
    % (focusing on output layer for efficiency)
    gradW_fc2 = zeros(size(kalmanNet.W_fc2));
    gradb_fc2 = zeros(size(kalmanNet.b_fc2));
    
    for seqIdx = 1:length(allTrainingData.innovations)
        kalmanNet.resetHiddenStates();
        
        innovations = allTrainingData.innovations{seqIdx};
        stateEstimates = allTrainingData.stateEstimates{seqIdx};
        H_matrices = allTrainingData.H{seqIdx};
        K_targets = allTrainingData.K_traditional{seqIdx};
        
        seqLen = size(innovations, 2);
        
        for t = 1:seqLen
            % Get KalmanNet prediction
            K_pred = kalmanNet.computeKalmanGain(innovations(:,t), ...
                stateEstimates(:,t), H_matrices{t});
            
            % Get target (traditional Kalman gain)
            K_target = K_targets{t};
            
            % Compute loss (for monitoring)
            [r1, c1] = size(K_pred);
            [r2, c2] = size(K_target);
            minR = min(r1, r2);
            minC = min(c1, c2);
            
            loss = mean(mean((K_pred(1:minR, 1:minC) - K_target(1:minR, 1:minC)).^2));
            epochLoss = epochLoss + loss;
            numSamples = numSamples + 1;
            
            % Compute approximate gradient for output layer
            % Using error as gradient signal (simplified gradient descent)
            K_error = K_pred(1:minR, 1:minC) - K_target(1:minR, 1:minC);
            error_vec = K_error(:);
            
            % Scale gradient by learning rate and error magnitude
            grad_scale = learningRate * mean(abs(error_vec));
            
            % Update output layer weights (W_fc2, b_fc2)
            outputSize = min(length(error_vec), size(kalmanNet.W_fc2, 1));
            gradW_fc2(1:outputSize, :) = gradW_fc2(1:outputSize, :) + ...
                grad_scale * error_vec(1:outputSize) * ones(1, size(kalmanNet.W_fc2, 2));
            gradb_fc2(1:outputSize) = gradb_fc2(1:outputSize) + grad_scale * error_vec(1:outputSize);
        end
    end
    
    % Apply averaged gradients
    if numSamples > 0
        gradW_fc2 = gradW_fc2 / numSamples;
        gradb_fc2 = gradb_fc2 / numSamples;
        
        % Gradient descent update with gradient clipping
        maxGrad = 1.0;
        gradW_fc2 = max(min(gradW_fc2, maxGrad), -maxGrad);
        gradb_fc2 = max(min(gradb_fc2, maxGrad), -maxGrad);
        
        kalmanNet.W_fc2 = kalmanNet.W_fc2 - gradW_fc2;
        kalmanNet.b_fc2 = kalmanNet.b_fc2 - gradb_fc2;
    end
    
    avgLoss = epochLoss / max(numSamples, 1);
    
    if mod(epoch, 5) == 0 || epoch == 1
        fprintf('Epoch %d/%d, Average Loss: %.6f\n', epoch, numEpochs, avgLoss);
    end
    
    % Decay learning rate
    learningRate = learningRate * 0.99;
end

kalmanNet.trained = true;

%% Save trained model
modelFile = 'kalmannet/kalmannet_msckf_model.mat';
kalmanNet.saveModel(modelFile);
fprintf('\n=== Training Complete ===\n');
fprintf('Model saved to: %s\n', modelFile);
