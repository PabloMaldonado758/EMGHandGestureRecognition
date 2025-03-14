clc; clear; close all;

% Define the dataset path
data_path = fullfile('data', 'datasets', 'Mes0');

% Check if the dataset directory exists
if exist(data_path, 'dir') == 0
    error('ERROR: The directory %s does not exist. Verify the path.', data_path);
end

% Get the list of users
users = dir(data_path);
users = users([users.isdir] & ~startsWith({users.name}, '.'));

% Define gesture labels
gestures = {'fist', 'open', 'pinch', 'relax', 'waveIn', 'waveOut'};

% Initialize structure to store EMG data
data_EMG = struct();

% Low-pass filter parameters
fs = 1000;  
fc = 20;    
[b, a] = butter(4, fc/(fs/2), 'low'); 

% Iterate over each user
for u = 1:length(users)
    user_name = users(u).name;
    user_path = fullfile(data_path, user_name);
    
    fprintf('Processing user: %s\n', user_name);
    
    % Iterate over each gesture
    for g = 1:length(gestures)
        gesture_name = gestures{g};
        file_path = fullfile(user_path, [gesture_name, '.mat']);
        
        if exist(file_path, 'file')
            mat_data = load(file_path);
            
            if isfield(mat_data, 'reps')
                reps = mat_data.reps;
                
                if isfield(reps, gesture_name)
                    gesture_data = reps.(gesture_name);
                    
                    if isfield(gesture_data, 'data')
                        data_struct = gesture_data.data{1}; 

                        % Extract EMG data and apply low-pass filter
                        if isfield(data_struct, 'emg')
                            emg_data = data_struct.emg;
                            
                            for ch = 1:size(emg_data,2)
                                emg_data(:, ch) = filtfilt(b, a, emg_data(:, ch));
                            end
                            
                            % Store filtered EMG data
                            data_EMG.(user_name).(gesture_name) = emg_data;
                            fprintf('Loaded: %s for %s\n', gesture_name, user_name);
                        else
                            fprintf('No EMG data found in %s\n', file_path);
                        end
                    end
                end
            end
        else
            fprintf('File not found: %s\n', file_path);
        end
    end
end

% Save processed data
save(fullfile('src', 'data_EMG.mat'), 'data_EMG');
fprintf('Data loading completed. Saved in src/data_EMG.mat\n');


