classdef KalmanNetNN < handle
    % KalmanNetNN: Neural network-based Kalman gain computation
    % This class implements the KalmanNet architecture that learns the Kalman
    % gain using GRU (Gated Recurrent Unit) networks, replacing the traditional
    % Kalman gain computation K = P*H'/(H*P*H' + R)
    %
    % Reference: "KalmanNet: Neural Network Aided Kalman Filtering for 
    %            Partially Known Dynamics" - Revach et al.
    
    properties
        % Network dimensions
        stateDim        % State dimension
        obsDim          % Observation dimension
        hiddenDim       % Hidden layer dimension
        
        % GRU parameters for Feature Extraction
        % GRU 1: Process observation difference
        W_z1            % Update gate weights
        U_z1            % Update gate recurrent weights
        b_z1            % Update gate bias
        W_r1            % Reset gate weights
        U_r1            % Reset gate recurrent weights
        b_r1            % Reset gate bias
        W_h1            % Hidden state weights
        U_h1            % Hidden state recurrent weights
        b_h1            % Hidden state bias
        
        % GRU 2: Process state estimate
        W_z2
        U_z2
        b_z2
        W_r2
        U_r2
        b_r2
        W_h2
        U_h2
        b_h2
        
        % Fully connected layers for Kalman Gain
        W_fc1           % First fully connected layer
        b_fc1
        W_fc2           % Second fully connected layer (outputs Kalman gain)
        b_fc2
        
        % Hidden states
        h1              % Hidden state for GRU 1
        h2              % Hidden state for GRU 2
        
        % Training parameters
        learningRate
        trained         % Flag indicating if network is trained
        
        % State and observation history for training
        prevObsDiff     % Previous observation difference
        prevStateEst    % Previous state estimate
    end
    
    methods
        function obj = KalmanNetNN(stateDim, obsDim, hiddenDim)
            % Constructor: Initialize KalmanNet with given dimensions
            % Inputs:
            %   stateDim: Dimension of state vector
            %   obsDim: Dimension of observation vector
            %   hiddenDim: Dimension of hidden layers
            
            obj.stateDim = stateDim;
            obj.obsDim = obsDim;
            obj.hiddenDim = hiddenDim;
            obj.learningRate = 0.001;
            obj.trained = false;
            
            % Initialize network parameters
            obj.initializeWeights();
            
            % Initialize hidden states
            obj.resetHiddenStates();
        end
        
        function initializeWeights(obj)
            % Initialize all network weights using Xavier initialization
            
            % GRU 1 parameters (processes observation difference)
            inputDim1 = obj.obsDim;
            obj.W_z1 = obj.xavierInit(obj.hiddenDim, inputDim1);
            obj.U_z1 = obj.xavierInit(obj.hiddenDim, obj.hiddenDim);
            obj.b_z1 = zeros(obj.hiddenDim, 1);
            
            obj.W_r1 = obj.xavierInit(obj.hiddenDim, inputDim1);
            obj.U_r1 = obj.xavierInit(obj.hiddenDim, obj.hiddenDim);
            obj.b_r1 = zeros(obj.hiddenDim, 1);
            
            obj.W_h1 = obj.xavierInit(obj.hiddenDim, inputDim1);
            obj.U_h1 = obj.xavierInit(obj.hiddenDim, obj.hiddenDim);
            obj.b_h1 = zeros(obj.hiddenDim, 1);
            
            % GRU 2 parameters (processes state estimate difference)
            inputDim2 = obj.stateDim;
            obj.W_z2 = obj.xavierInit(obj.hiddenDim, inputDim2);
            obj.U_z2 = obj.xavierInit(obj.hiddenDim, obj.hiddenDim);
            obj.b_z2 = zeros(obj.hiddenDim, 1);
            
            obj.W_r2 = obj.xavierInit(obj.hiddenDim, inputDim2);
            obj.U_r2 = obj.xavierInit(obj.hiddenDim, obj.hiddenDim);
            obj.b_r2 = zeros(obj.hiddenDim, 1);
            
            obj.W_h2 = obj.xavierInit(obj.hiddenDim, inputDim2);
            obj.U_h2 = obj.xavierInit(obj.hiddenDim, obj.hiddenDim);
            obj.b_h2 = zeros(obj.hiddenDim, 1);
            
            % Fully connected layers for Kalman Gain output
            fcInputDim = 2 * obj.hiddenDim;
            fcOutputDim = obj.stateDim * obj.obsDim;
            
            obj.W_fc1 = obj.xavierInit(obj.hiddenDim, fcInputDim);
            obj.b_fc1 = zeros(obj.hiddenDim, 1);
            
            obj.W_fc2 = obj.xavierInit(fcOutputDim, obj.hiddenDim);
            obj.b_fc2 = zeros(fcOutputDim, 1);
        end
        
        function W = xavierInit(obj, rows, cols)
            % Xavier/Glorot initialization
            stddev = sqrt(2.0 / (rows + cols));
            W = stddev * randn(rows, cols);
        end
        
        function resetHiddenStates(obj)
            % Reset GRU hidden states
            obj.h1 = zeros(obj.hiddenDim, 1);
            obj.h2 = zeros(obj.hiddenDim, 1);
            obj.prevObsDiff = zeros(obj.obsDim, 1);
            obj.prevStateEst = zeros(obj.stateDim, 1);
        end
        
        function K = computeKalmanGain(obj, innovation, stateEstimate, H)
            % Compute Kalman Gain using neural network
            % Inputs:
            %   innovation: z - H*x_pred (observation residual)
            %   stateEstimate: Current state estimate
            %   H: Observation matrix
            % Output:
            %   K: Kalman gain matrix (stateDim x obsDim)
            
            % Ensure inputs are column vectors
            if size(innovation, 2) > 1
                innovation = innovation(:);
            end
            if size(stateEstimate, 2) > 1
                stateEstimate = stateEstimate(:);
            end
            
            % Handle dimension mismatch
            actualObsDim = length(innovation);
            actualStateDim = length(stateEstimate);
            
            % Pad or truncate innovation to match expected obsDim
            if actualObsDim < obj.obsDim
                innovation = [innovation; zeros(obj.obsDim - actualObsDim, 1)];
            elseif actualObsDim > obj.obsDim
                innovation = innovation(1:obj.obsDim);
            end
            
            % Pad or truncate state estimate to match expected stateDim
            if actualStateDim < obj.stateDim
                stateEstimate = [stateEstimate; zeros(obj.stateDim - actualStateDim, 1)];
            elseif actualStateDim > obj.stateDim
                stateEstimate = stateEstimate(1:obj.stateDim);
            end
            
            % GRU 1: Process innovation (observation difference)
            obj.h1 = obj.gruForward(innovation, obj.h1, ...
                obj.W_z1, obj.U_z1, obj.b_z1, ...
                obj.W_r1, obj.U_r1, obj.b_r1, ...
                obj.W_h1, obj.U_h1, obj.b_h1);
            
            % GRU 2: Process state estimate
            obj.h2 = obj.gruForward(stateEstimate, obj.h2, ...
                obj.W_z2, obj.U_z2, obj.b_z2, ...
                obj.W_r2, obj.U_r2, obj.b_r2, ...
                obj.W_h2, obj.U_h2, obj.b_h2);
            
            % Concatenate hidden states
            combinedHidden = [obj.h1; obj.h2];
            
            % Fully connected layers
            fc1_out = tanh(obj.W_fc1 * combinedHidden + obj.b_fc1);
            fc2_out = obj.W_fc2 * fc1_out + obj.b_fc2;
            
            % Reshape to Kalman gain matrix
            K = reshape(fc2_out, [obj.stateDim, obj.obsDim]);
            
            % Handle actual output dimensions
            if actualStateDim ~= obj.stateDim || actualObsDim ~= obj.obsDim
                K = K(1:min(actualStateDim, obj.stateDim), 1:min(actualObsDim, obj.obsDim));
                % Pad if necessary
                if actualStateDim > obj.stateDim || actualObsDim > obj.obsDim
                    K_full = zeros(actualStateDim, actualObsDim);
                    K_full(1:size(K,1), 1:size(K,2)) = K;
                    K = K_full;
                end
            end
            
            % Store for next iteration
            obj.prevObsDiff = innovation;
            obj.prevStateEst = stateEstimate;
        end
        
        function h_new = gruForward(obj, x, h_prev, W_z, U_z, b_z, W_r, U_r, b_r, W_h, U_h, b_h)
            % GRU forward pass
            % Update gate
            z = obj.sigmoid(W_z * x + U_z * h_prev + b_z);
            % Reset gate
            r = obj.sigmoid(W_r * x + U_r * h_prev + b_r);
            % Candidate hidden state
            h_tilde = tanh(W_h * x + U_h * (r .* h_prev) + b_h);
            % New hidden state
            h_new = (1 - z) .* h_prev + z .* h_tilde;
        end
        
        function y = sigmoid(obj, x)
            % Sigmoid activation function
            y = 1 ./ (1 + exp(-x));
        end
        
        function train(obj, trainingData, numEpochs, batchSize)
            % Train the KalmanNet on collected data
            % Inputs:
            %   trainingData: Structure with fields:
            %       .innovations: Cell array of innovation sequences
            %       .stateEstimates: Cell array of state estimate sequences
            %       .trueStates: Cell array of true state sequences
            %       .H: Cell array of observation matrices
            %   numEpochs: Number of training epochs
            %   batchSize: Batch size for training
            
            if nargin < 3
                numEpochs = 100;
            end
            if nargin < 4
                batchSize = 32;
            end
            
            numSequences = length(trainingData.innovations);
            
            for epoch = 1:numEpochs
                totalLoss = 0;
                
                % Shuffle sequences
                randIdx = randperm(numSequences);
                
                for seqIdx = 1:numSequences
                    idx = randIdx(seqIdx);
                    
                    % Reset hidden states for each sequence
                    obj.resetHiddenStates();
                    
                    innovations = trainingData.innovations{idx};
                    stateEstimates = trainingData.stateEstimates{idx};
                    trueStates = trainingData.trueStates{idx};
                    H_matrices = trainingData.H{idx};
                    
                    seqLen = size(innovations, 2);
                    seqLoss = 0;
                    
                    for t = 1:seqLen
                        % Forward pass
                        K = obj.computeKalmanGain(innovations(:,t), ...
                            stateEstimates(:,t), H_matrices{t});
                        
                        % Compute predicted state update
                        deltaX = K * innovations(:,t);
                        
                        % Loss: MSE between predicted update and optimal update
                        if t < seqLen
                            optimalUpdate = trueStates(:,t+1) - stateEstimates(:,t);
                            loss = mean((deltaX - optimalUpdate).^2);
                            seqLoss = seqLoss + loss;
                        end
                    end
                    
                    totalLoss = totalLoss + seqLoss / seqLen;
                end
                
                avgLoss = totalLoss / numSequences;
                
                if mod(epoch, 10) == 0
                    fprintf('Epoch %d/%d, Average Loss: %.6f\n', epoch, numEpochs, avgLoss);
                end
            end
            
            obj.trained = true;
            fprintf('Training complete.\n');
        end
        
        function saveModel(obj, filename)
            % Save model parameters to file
            modelParams.stateDim = obj.stateDim;
            modelParams.obsDim = obj.obsDim;
            modelParams.hiddenDim = obj.hiddenDim;
            
            % GRU 1 parameters
            modelParams.W_z1 = obj.W_z1;
            modelParams.U_z1 = obj.U_z1;
            modelParams.b_z1 = obj.b_z1;
            modelParams.W_r1 = obj.W_r1;
            modelParams.U_r1 = obj.U_r1;
            modelParams.b_r1 = obj.b_r1;
            modelParams.W_h1 = obj.W_h1;
            modelParams.U_h1 = obj.U_h1;
            modelParams.b_h1 = obj.b_h1;
            
            % GRU 2 parameters
            modelParams.W_z2 = obj.W_z2;
            modelParams.U_z2 = obj.U_z2;
            modelParams.b_z2 = obj.b_z2;
            modelParams.W_r2 = obj.W_r2;
            modelParams.U_r2 = obj.U_r2;
            modelParams.b_r2 = obj.b_r2;
            modelParams.W_h2 = obj.W_h2;
            modelParams.U_h2 = obj.U_h2;
            modelParams.b_h2 = obj.b_h2;
            
            % FC layers
            modelParams.W_fc1 = obj.W_fc1;
            modelParams.b_fc1 = obj.b_fc1;
            modelParams.W_fc2 = obj.W_fc2;
            modelParams.b_fc2 = obj.b_fc2;
            
            modelParams.trained = obj.trained;
            
            save(filename, 'modelParams');
            fprintf('Model saved to %s\n', filename);
        end
        
        function loadModel(obj, filename)
            % Load model parameters from file
            % Input:
            %   filename: Path to the model file (.mat)
            
            % Check if file exists
            if ~exist(filename, 'file')
                error('KalmanNetNN:FileNotFound', ...
                    'Model file not found: %s\nPlease train the model first using trainKalmanNet.m', filename);
            end
            
            try
                data = load(filename);
                
                if ~isfield(data, 'modelParams')
                    error('KalmanNetNN:InvalidFormat', ...
                        'Invalid model file format: missing modelParams structure');
                end
                
                modelParams = data.modelParams;
                
                % Validate required fields
                requiredFields = {'stateDim', 'obsDim', 'hiddenDim', 'W_z1', 'W_fc2'};
                for i = 1:length(requiredFields)
                    if ~isfield(modelParams, requiredFields{i})
                        error('KalmanNetNN:InvalidFormat', ...
                            'Invalid model file: missing field %s', requiredFields{i});
                    end
                end
                
                obj.stateDim = modelParams.stateDim;
                obj.obsDim = modelParams.obsDim;
                obj.hiddenDim = modelParams.hiddenDim;
                
                % GRU 1 parameters
                obj.W_z1 = modelParams.W_z1;
                obj.U_z1 = modelParams.U_z1;
                obj.b_z1 = modelParams.b_z1;
                obj.W_r1 = modelParams.W_r1;
                obj.U_r1 = modelParams.U_r1;
                obj.b_r1 = modelParams.b_r1;
                obj.W_h1 = modelParams.W_h1;
                obj.U_h1 = modelParams.U_h1;
                obj.b_h1 = modelParams.b_h1;
                
                % GRU 2 parameters
                obj.W_z2 = modelParams.W_z2;
                obj.U_z2 = modelParams.U_z2;
                obj.b_z2 = modelParams.b_z2;
                obj.W_r2 = modelParams.W_r2;
                obj.U_r2 = modelParams.U_r2;
                obj.b_r2 = modelParams.b_r2;
                obj.W_h2 = modelParams.W_h2;
                obj.U_h2 = modelParams.U_h2;
                obj.b_h2 = modelParams.b_h2;
                
                % FC layers
                obj.W_fc1 = modelParams.W_fc1;
                obj.b_fc1 = modelParams.b_fc1;
                obj.W_fc2 = modelParams.W_fc2;
                obj.b_fc2 = modelParams.b_fc2;
                
                obj.trained = modelParams.trained;
                
                obj.resetHiddenStates();
                
                fprintf('Model loaded from %s\n', filename);
                
            catch ME
                if strcmp(ME.identifier, 'KalmanNetNN:FileNotFound') || ...
                   strcmp(ME.identifier, 'KalmanNetNN:InvalidFormat')
                    rethrow(ME);
                else
                    error('KalmanNetNN:LoadError', ...
                        'Error loading model from %s: %s', filename, ME.message);
                end
            end
        end
    end
end
