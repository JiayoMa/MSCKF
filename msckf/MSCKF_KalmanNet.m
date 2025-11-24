%% ============================Notation============================ %%
% X_sub_super
% q_ToFrom
% p_ofWhat_expressedInWhatFrame
%
% MSCKF_KalmanNet: MSCKF implementation using KalmanNet for Kalman gain
% This version replaces the traditional EKF Kalman gain computation with
% a neural network-based approach (KalmanNet)


%% =============================Setup============================== %%
clear;
close all;
clc;
addpath('utils');
addpath('kalmannet');

tic
dataDir = '../datasets';

% fileName = 'dataset3';
% fileName = 'dataset3_fresh_10noisy';
% fileName = 'dataset3_fresh_10lessnoisy';
% fileName = 'dataset3_fresh_20lessnoisy';
% fileName = 'dataset3_fresh_40lessnoisy';
% fileName = 'dataset3_fresh_60lessnoisy';
% fileName = 'dataset3_fresh_80lessnoisy';
% fileName = 'dataset3_fresh_100lessnoisy';
% fileName = 'dataset3_fresh_500lessnoisy';
% fileName = '2011_09_26_drive_0035_sync_KLT';
% fileName = '2011_09_26_drive_0005_sync_KLT';
% fileName = '2011_09_30_drive_0020_sync_KLT';
% fileName = '2011_09_26_drive_0027_sync_KLT';
% fileName = '2011_09_30_drive_0020_sync_KLT'; kStart = 2; kEnd = 900;

% Good KITTI runs
 fileName = '2011_09_26_drive_0001_sync_KLT'; kStart = 2; kEnd = 98;
%fileName = '2011_09_26_drive_0036_sync_KLT'; kStart = 2; kEnd = 239;
% fileName = '2011_09_26_drive_0051_sync_KLT'; kStart = 2; kEnd = 114;
% fileName = '2011_09_26_drive_0095_sync_KLT'; kStart = 2; kEnd = 139;

%Step 1: Load dataset
load(sprintf('%s/%s.mat',dataDir,fileName));

% r_i_vk_i = p_vi_i;

%Dataset window bounds
% kStart = 2; kEnd = 177;
% kStart = 1215; kEnd = 1715;

%Set constant
numLandmarks = size(y_k_j,3);

%Set up the camera parameters
camera.c_u      = cu;                   % Principal point [u pixels]
camera.c_v      = cv;                   % Principal point [v pixels]
camera.f_u      = fu;                   % Focal length [u pixels]
camera.f_v      = fv;                   % Focal length [v pixels]
camera.b        = b;                    % Stereo baseline [m]
camera.q_CI     = rotMatToQuat(C_c_v);  % 4x1 IMU-to-Camera rotation quaternion
camera.p_C_I    = rho_v_c_v;            % 3x1 Camera position in IMU frame

%Set up the noise parameters
y_var = 11^2 * ones(1,4);               % pixel coord var 121
noiseParams.u_var_prime = y_var(1)/camera.f_u^2;
noiseParams.v_var_prime = y_var(2)/camera.f_v^2;

%Step 2: Initialize IMU prediction covariance and noise covariance matrix
%Step 2.1: Initialize noise covariance matrix
w_var = 4e-2 * ones(1,3);              % rot vel var0.04
v_var = 4e-2 * ones(1,3);              % lin vel var
dbg_var = 1e-6 * ones(1,3);            % gyro bias change var
dbv_var = 1e-6 * ones(1,3);            % vel bias change var
noiseParams.Q_imu = diag([w_var, dbg_var, v_var, dbv_var]);

%Step 2.2: Initialize state prediction covariance matrix
q_var_init = 1e-6 * ones(1,3);         % init rot var
p_var_init = 1e-6 * ones(1,3);         % init pos var
bg_var_init = 1e-6 * ones(1,3);        % init gyro bias var
bv_var_init = 1e-6 * ones(1,3);        % init vel bias var
noiseParams.initialIMUCovar = diag([q_var_init, bg_var_init, bv_var_init, p_var_init]);
   
% MSCKF parameters
msckfParams.minTrackLength = 10;        % Set to inf to dead-reckon only
msckfParams.maxTrackLength = Inf;      % Set to inf to wait for features to go out of view
msckfParams.maxGNCostNorm  = 1e-2;     % Set to inf to allow any triangulation, no matter how bad
msckfParams.minRCOND       = 1e-12;
msckfParams.doNullSpaceTrick = true;
msckfParams.doQRdecomp = true;

%% ========================KalmanNet Setup========================= %%
% Initialize KalmanNet neural network
% State dimension: 12 (IMU) + 6*N (camera states)
% We use a base dimension and let KalmanNet adapt
baseStateDim = 12 + 6 * 10;  % Initial estimate: 12 IMU + 6*10 camera states
baseObsDim = 20;             % Initial observation dimension estimate
hiddenDim = 64;              % Hidden layer dimension

% Create KalmanNet instance
kalmanNet = KalmanNetNN(baseStateDim, baseObsDim, hiddenDim);

% KalmanNet parameters
msckfParams.useKalmanNet = true;      % Flag to use KalmanNet
msckfParams.kalmanNetBlend = 1.0;     % Blend factor (1.0 = full KalmanNet)

% Check if pre-trained model exists
kalmanNetModelFile = 'kalmannet/kalmannet_msckf_model.mat';
if exist(kalmanNetModelFile, 'file')
    fprintf('Loading pre-trained KalmanNet model...\n');
    kalmanNet.loadModel(kalmanNetModelFile);
else
    fprintf('Using randomly initialized KalmanNet (no pre-trained model found).\n');
    fprintf('Consider training the model using trainKalmanNet.m for better performance.\n');
end


% IMU state for plotting etc. Structures indexed in a cell array
imuStates = cell(1,numel(t));
prunedStates = {};

% imuStates{k}.q_IG         4x1 Global to IMU rotation quaternion
% imuStates{k}.p_I_G        3x1 IMU Position in the Global frame
% imuStates{k}.b_g          3x1 Gyro bias
% imuStates{k}.b_v          3x1 Velocity bias
% imuStates{k}.covar        12x12 IMU state covariance

%msckfState.imuState
%msckfState.imuCovar
%msckfState.camCovar
%msckfState.imuCamCovar
%msckfState.camStates


% Measurements as structures all indexed in a cell array
dT = [0, diff(t)];
measurements = cell(1,numel(t));

% Important: Because we're idealizing our pixel measurements and the
% idealized measurements could legitimately be -1, replace our invalid
% measurement flag with NaN
y_k_j(y_k_j == -1) = NaN;

%Step 3: Construct measurement data and reference values
for state_k = kStart:kEnd 
    measurements{state_k}.dT    = dT(state_k);                      % sampling times
    measurements{state_k}.y     = squeeze(y_k_j(1:2,state_k,:));    % left camera only
    measurements{state_k}.omega = w_vk_vk_i(:,state_k);             % ang vel
    measurements{state_k}.v     = v_vk_vk_i(:,state_k);             % lin vel
    
    %Idealize measurements
    validMeas = ~isnan(measurements{state_k}.y(1,:));
    measurements{state_k}.y(1,validMeas) = (measurements{state_k}.y(1,validMeas) - camera.c_u)/camera.f_u;
    measurements{state_k}.y(2,validMeas) = (measurements{state_k}.y(2,validMeas) - camera.c_v)/camera.f_v;
    
    %Ground Truth
    q_IG = rotMatToQuat(axisAngleToRotMat(theta_vk_i(:,state_k)));
    p_I_G = r_i_vk_i(:,state_k);
    
    groundTruthStates{state_k}.imuState.q_IG = q_IG;
    groundTruthStates{state_k}.imuState.p_I_G = p_I_G;
    
    % Compute camera pose from current IMU pose
    C_IG = quatToRotMat(q_IG);
    q_CG = quatLeftComp(camera.q_CI) * q_IG;
    p_C_G = p_I_G + C_IG' * camera.p_C_I;
    
    groundTruthStates{state_k}.camState.q_CG = q_CG;
    groundTruthStates{state_k}.camState.p_C_G = p_C_G;
    
end


%Struct used to keep track of features
featureTracks = {};
trackedFeatureIds = [];

% featureTracks = {track1, track2, ...}
% track.featureId 
% track.observations



%% ==========================Initial State======================== %%
%Use ground truth for first state and initialize feature tracks with
%feature observations
%Use ground truth for the first state

%Step 4: Initialize MSCKF
%Step 4.1: Get first quaternion reference value as IMU state initial value
firstImuState.q_IG = rotMatToQuat(axisAngleToRotMat(theta_vk_i(:,kStart)));
%Step 4.2: Get first position reference value as position initial value
firstImuState.p_I_G = r_i_vk_i(:,kStart);

%Step 4.3: Initialize msckf state, currently only IMU state, initialize tracked feature points
[msckfState, featureTracks, trackedFeatureIds] = initializeMSCKF(firstImuState, measurements{kStart}, camera, kStart, noiseParams);
%Update IMU history state from MSCKF state
imuStates = updateStateHistory(imuStates, msckfState, camera, kStart);
%For plotting use
msckfState_imuOnly{kStart} = msckfState;

%% ============================MAIN LOOP========================== %%

numFeatureTracksResidualized = 0;
map = [];

% Data collection for KalmanNet training
kalmanNetTrainingData.innovations = {};
kalmanNetTrainingData.stateEstimates = {};
kalmanNetTrainingData.trueStates = {};
kalmanNetTrainingData.H = {};

%Main loop over all frames
for state_k = kStart:(kEnd-1)
    fprintf('state_k = %4d\n', state_k);
    
    %% ==========================STATE PROPAGATION======================== %%
    %Step 5: MSCKF prediction update - state propagation and covariance update
    msckfState = propagateMsckfStateAndCovar(msckfState, measurements{state_k}, noiseParams);
    msckfState_imuOnly{state_k+1} = propagateMsckfStateAndCovar(msckfState_imuOnly{state_k}, measurements{state_k}, noiseParams);
    %Step 6: Augment state with camera state, compute Jacobian, update covariance matrix
    msckfState = augmentState(msckfState, camera, state_k+1);
    %% ==========================FEATURE TRACKING======================== %%
    % Add observations to the feature tracks, or initialize a new one
    % If an observation is -1, add the track to featureTracksToResidualize
    featureTracksToResidualize = {};
    
    %Step 7: Process current frame feature points, update featureTracks
    
    %Process all features in current frame
    for featureId = 1:numLandmarks
        %IMPORTANT: state_k + 1 not state_k
        meas_k = measurements{state_k+1}.y(:, featureId);
        
        outOfView = isnan(meas_k(1,1));
        
        %Step 7.1: Check if feature point is being tracked
        if ismember(featureId, trackedFeatureIds)

            %Step 7.2: If feature is in view, add observation to featureTracks
            if ~outOfView
                featureTracks{trackedFeatureIds == featureId}.observations(:, end+1) = meas_k;
                msckfState.camStates{end}.trackedFeatureIds(end+1) = featureId;
            end
            
            %Step 7.3: If feature is out of view or has enough observations
            track = featureTracks{trackedFeatureIds == featureId};
            
            if outOfView ...
                    || size(track.observations, 2) >= msckfParams.maxTrackLength ...
                    || state_k+1 == kEnd
                                
                [msckfState, camStates, camStateIndices] = removeTrackedFeature(msckfState, featureId);
                
                if length(camStates) >= msckfParams.minTrackLength
                    track.camStates = camStates;
                    track.camStateIndices = camStateIndices;
                    featureTracksToResidualize{end+1} = track;
                end
               
                featureTracks = featureTracks(trackedFeatureIds ~= featureId);
                trackedFeatureIds(trackedFeatureIds == featureId) = []; 
            end
        
        %Step 7.4: Track new feature points
        elseif ~outOfView && state_k+1 < kEnd
            track.featureId = featureId;
            track.observations = meas_k;
            featureTracks{end+1} = track;
            trackedFeatureIds(end+1) = featureId;

            msckfState.camStates{end}.trackedFeatureIds(end+1) = featureId;
        end
    end
     
    %Step 8: MSCKF measurement update using KalmanNet
    %% ==========================FEATURE RESIDUAL CORRECTIONS======================== %%
    if ~isempty(featureTracksToResidualize)
        H_o = [];
        r_o = [];
        R_o = [];
        %Step 8.1: Estimate 3D position through triangulation
        for f_i = 1:length(featureTracksToResidualize)
            track = featureTracksToResidualize{f_i};     
            [p_f_G, Jcost, RCOND] = calcGNPosEst(track.camStates, track.observations, noiseParams);
            
            nObs = size(track.observations,2);
            JcostNorm = Jcost / nObs^2;
            fprintf('Jcost = %f | JcostNorm = %f | RCOND = %f\n',...
                Jcost, JcostNorm,RCOND);
            
            if JcostNorm > msckfParams.maxGNCostNorm ...
                    || RCOND < msckfParams.minRCOND
                
                continue;  % Skip this feature track and try next one
            else
                map(:,end+1) = p_f_G;
                numFeatureTracksResidualized = numFeatureTracksResidualized + 1;
                fprintf('Using new feature track with %d observations. Total track count = %d.\n',...
                    nObs, numFeatureTracksResidualized);
            end
            %Step 8.2: Calculate residual and observation Jacobian
            [r_j] = calcResidual(p_f_G, track.camStates, track.observations);
            R_j = diag(repmat([noiseParams.u_var_prime, noiseParams.v_var_prime], [1, numel(r_j)/2]));
            [H_o_j, A_j, H_x_j] = calcHoj(p_f_G, msckfState, track.camStateIndices);

            % Stacked residuals and friends
            if msckfParams.doNullSpaceTrick
                H_o = [H_o; H_o_j];

                if ~isempty(A_j)
                    r_o_j = A_j' * r_j;
                    r_o = [r_o ; r_o_j];

                    R_o_j = A_j' * R_j * A_j;
                    R_o(end+1 : end+size(R_o_j,1), end+1 : end+size(R_o_j,2)) = R_o_j;
                end
                
            else
                H_o = [H_o; H_x_j];
                r_o = [r_o; r_j];
                R_o(end+1 : end+size(R_j,1), end+1 : end+size(R_j,2)) = R_j;
            end
        end
        
        if ~isempty(r_o)
            % Put residuals into their final update-worthy form
            if msckfParams.doQRdecomp
                [T_H, Q_1] = calcTH(H_o);
                r_n = Q_1' * r_o;
                R_n = Q_1' * R_o * Q_1;
            else
                T_H = H_o;
                r_n = r_o;
                R_n = R_o;
            end           
            %Step 8.3: Calculate Kalman gain using KalmanNet
            % Build MSCKF covariance matrix
            P = [msckfState.imuCovar, msckfState.imuCamCovar;
                   msckfState.imuCamCovar', msckfState.camCovar];

            % Calculate Kalman gain - Using KalmanNet instead of traditional EKF
            stateDim = 12 + 6*size(msckfState.camStates,2);
            
            if msckfParams.useKalmanNet
                % Use KalmanNet to compute Kalman gain
                K = computeKalmanGainKalmanNet(kalmanNet, T_H, P, R_n, r_n, stateDim);
                
                % Optionally blend with traditional Kalman gain
                if msckfParams.kalmanNetBlend < 1.0
                    K_traditional = (P*T_H') / ( T_H*P*T_H' + R_n );
                    K = msckfParams.kalmanNetBlend * K + (1 - msckfParams.kalmanNetBlend) * K_traditional;
                end
            else
                % Traditional Kalman gain computation (fallback)
                K = (P*T_H') / ( T_H*P*T_H' + R_n );
            end

            % State correction
            deltaX = K * r_n;
            %Step 8.4: Update state
            msckfState = updateState(msckfState, deltaX);

            % Covariance correction
            %Step 8.5: Update covariance
            tempMat = (eye(12 + 6*size(msckfState.camStates,2)) - K*T_H);

            P_corrected = tempMat * P * tempMat' + K * R_n * K';

            msckfState.imuCovar = P_corrected(1:12,1:12);
            msckfState.camCovar = P_corrected(13:end,13:end);
            msckfState.imuCamCovar = P_corrected(1:12, 13:end);
           
        end
        
    end
    
        %% ==========================STATE HISTORY======================== %%
        %Step 9: Update history state
        imuStates = updateStateHistory(imuStates, msckfState, camera, state_k+1);
        
        
        %% ==========================STATE PRUNING======================== %%
        %Step 10: Prune states
        [msckfState, deletedCamStates] = pruneStates(msckfState);

        if ~isempty(deletedCamStates)
            prunedStates(end+1:end+length(deletedCamStates)) = deletedCamStates;
        end    
        
        plot_traj;
end %for state_K = ...

toc


%% ==========================PLOT ERRORS======================== %%
kNum = length(prunedStates);
p_C_G_est = NaN(3, kNum);
p_I_G_imu = NaN(3, kNum);
p_C_G_imu = NaN(3, kNum);
p_C_G_GT = NaN(3, kNum);
theta_CG_err = NaN(3,kNum);
theta_CG_err_imu = NaN(3,kNum);
err_sigma = NaN(6,kNum); % cam state is ordered as [rot, trans]
err_sigma_imu = NaN(6,kNum);
% 
tPlot = NaN(1, kNum);
% 
for k = 1:kNum
    state_k = prunedStates{k}.state_k;
    
    p_C_G_GT(:,k) = groundTruthStates{state_k}.camState.p_C_G;
    p_C_G_est(:,k) = prunedStates{k}.p_C_G;
    q_CG_est  = prunedStates{k}.q_CG;    
    
    theta_CG_err(:,k) = crossMatToVec( eye(3) ...
                    - quatToRotMat(q_CG_est) ...
                        * ( C_c_v * axisAngleToRotMat(theta_vk_i(:,kStart+k-1)) )' );
      
    err_sigma(:,k) = prunedStates{k}.sigma;
    imusig = sqrt(diag(msckfState_imuOnly{state_k}.imuCovar));
    err_sigma_imu(:,k) = imusig([1:3,10:12]);
    
    p_I_G_imu(:,k) = msckfState_imuOnly{state_k}.imuState.p_I_G;
    C_CG_est_imu = C_CI * quatToRotMat(msckfState_imuOnly{state_k}.imuState.q_IG);
    theta_CG_err_imu(:,k) = crossMatToVec( eye(3) ...
                    - C_CG_est_imu ...
                        * ( C_CI * axisAngleToRotMat(theta_vk_i(:,kStart+k-1)) )' );
                    
    tPlot(k) = t(state_k);
end

% p_I_G_GT = p_vi_i(:,kStart:kEnd);
p_I_G_GT = r_i_vk_i(:,kStart:kEnd);
p_C_G_GT = p_I_G_GT + repmat(rho_v_c_v,[1,size(p_I_G_GT,2)]);
p_C_G_imu = p_I_G_imu + repmat(rho_v_c_v,[1,size(p_I_G_imu,2)]);

rotLim = [-0.5 0.5];
transLim = [-0.5 0.5];

% Save estimates
msckf_trans_err = p_C_G_est - p_C_G_GT;
msckf_rot_err = theta_CG_err;
imu_trans_err = p_C_G_imu - p_C_G_GT;
imu_rot_err = theta_CG_err_imu;
save(sprintf('../KITTI Trials/msckf_kalmannet_%s', fileName));

armse_trans_msckf = mean(sqrt(sum(msckf_trans_err.^2, 1)/3));
armse_rot_msckf = mean(sqrt(sum(msckf_rot_err.^2, 1)/3));
final_trans_err_msckf = norm(msckf_trans_err(:,end));

armse_trans_imu = mean(sqrt(sum(imu_trans_err.^2, 1)/3));
armse_rot_imu = mean(sqrt(sum(imu_rot_err.^2, 1)/3));
final_trans_err_imu = norm(imu_trans_err(:,end));

fprintf('=== KalmanNet MSCKF Results ===\n');
fprintf('Trans ARMSE: IMU %f, MSCKF-KalmanNet %f\n',armse_trans_imu, armse_trans_msckf);
fprintf('Rot ARMSE: IMU %f, MSCKF-KalmanNet %f\n',armse_rot_imu, armse_rot_msckf);
fprintf('Final Trans Err: IMU %f, MSCKF-KalmanNet %f\n',final_trans_err_imu, final_trans_err_msckf);

% Translation Errors
figure
subplot(3,1,1)
plot(tPlot, p_C_G_est(1,:) - p_C_G_GT(1,:), 'LineWidth', 2)
hold on
plot(tPlot, 3*err_sigma(4,:), '--r')
plot(tPlot, -3*err_sigma(4,:), '--r')
% ylim(transLim)
xlim([tPlot(1) tPlot(end)])
title('Translational Error (KalmanNet MSCKF)')
ylabel('\delta r_x')


subplot(3,1,2)
plot(tPlot, p_C_G_est(2,:) - p_C_G_GT(2,:), 'LineWidth', 2)
hold on
plot(tPlot, 3*err_sigma(5,:), '--r')
plot(tPlot, -3*err_sigma(5,:), '--r')
% ylim(transLim)
xlim([tPlot(1) tPlot(end)])
ylabel('\delta r_y')

subplot(3,1,3)
plot(tPlot, p_C_G_est(3,:) - p_C_G_GT(3,:), 'LineWidth', 2)
hold on
plot(tPlot, 3*err_sigma(6,:), '--r')
plot(tPlot, -3*err_sigma(6,:), '--r')
% ylim(transLim)
xlim([tPlot(1) tPlot(end)])
ylabel('\delta r_z')
xlabel('t_k')

% Rotation Errors
figure
subplot(3,1,1)
plot(tPlot, theta_CG_err(1,:), 'LineWidth', 2)
hold on
plot(tPlot, 3*err_sigma(1,:), '--r')
plot(tPlot, -3*err_sigma(1,:), '--r')
ylim(rotLim)
xlim([tPlot(1) tPlot(end)])
title('Rotational Error (KalmanNet MSCKF)')
ylabel('\delta \theta_x')


subplot(3,1,2)
plot(tPlot, theta_CG_err(2,:), 'LineWidth', 2)
hold on
plot(tPlot, 3*err_sigma(2,:), '--r')
plot(tPlot, -3*err_sigma(2,:), '--r')
ylim(rotLim)
xlim([tPlot(1) tPlot(end)])
ylabel('\delta \theta_y')

subplot(3,1,3)
plot(tPlot, theta_CG_err(3,:), 'LineWidth', 2)
hold on
plot(tPlot, 3*err_sigma(3,:), '--r')
plot(tPlot, -3*err_sigma(3,:), '--r')
ylim(rotLim)
xlim([tPlot(1) tPlot(end)])
ylabel('\delta \theta_z')
xlabel('t_k')
