clc; clear; close all;

% Load processed EMG data from 'src' directory
load(fullfile('src', 'data_EMG.mat'));

% Define gesture labels and number of gestures
gesture_labels = {'fist', 'open', 'pinch', 'relax', 'waveIn', 'waveOut'};
num_gestures = length(gesture_labels);

% Initialize feature and label matrices
X = [];
y = [];

% Ensure 'src' directory exists
src_folder = 'src';
if ~exist(src_folder, 'dir')
    mkdir(src_folder);
end

% Iterate over each user
users = fieldnames(data_EMG);
for u = 1:length(users)
    user_name = users{u};
    
    % Iterate over each gesture
    for g = 1:num_gestures
        gesture_name = gesture_labels{g};
        
        % Check if the user has data for this gesture
        if isfield(data_EMG.(user_name), gesture_name)
            emg_data = data_EMG.(user_name).(gesture_name);
            
            % Ensure valid data samples
            if size(emg_data, 1) > 0
                % Normalize the sample size using interpolation
                target_length = 1000;
                [rows, cols] = size(emg_data);
                new_emg_data = zeros(target_length, cols);
                
                for ch = 1:cols
                    new_emg_data(:, ch) = interp1(linspace(0,1,rows), emg_data(:,ch), linspace(0,1,target_length), 'linear');
                end
                
                % Apply Z-score normalization
                new_emg_data = zscore(new_emg_data);

                % Extract statistical features from EMG signal
                features = [];
                for ch = 1:cols
                    ch_data = new_emg_data(:, ch);
                    features = [features, ...
                        mean(ch_data), std(ch_data), median(ch_data), max(ch_data), min(ch_data), ...
                        var(ch_data), range(ch_data), sum(abs(ch_data))];
                end

                % Store extracted features instead of raw signals
                flattened_data = features;

                % Ensure feature dimension consistency
                if isempty(X)
                    X = flattened_data;
                    y = g;
                elseif size(flattened_data, 2) == size(X, 2)
                    X = [X; flattened_data];
                    y = [y; g];
                else
                    fprintf('Warning: Dimension mismatch in %s - %s (%dx%d instead of 1x%d)\n', ...
                        user_name, gesture_name, size(flattened_data, 1), size(flattened_data, 2), size(X,2));
                end
            end
        end
    end
end

% Save processed dataset in 'src' directory
save(fullfile(src_folder, 'knn_dataset.mat'), 'X', 'y');
fprintf('Feature extraction completed. Data saved in src/knn_dataset.mat\n');
