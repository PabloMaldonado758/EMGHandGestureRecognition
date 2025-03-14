clc; clear; close all;

% Load the dataset with extracted features
load(fullfile('src', 'knn_dataset.mat'));

% Normalize data using Z-score
X = zscore(X);

% Set random seed for reproducibility and split data (80% training, 20% testing)
rng(42);
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Apply PCA to reduce dimensions before training KNN
[coeff, X_train_pca, ~, ~, explained] = pca(X_train);
variance_threshold = 95;
explained_variance = cumsum(explained);
num_components = find(explained_variance >= variance_threshold, 1);
X_train_pca = X_train_pca(:, 1:num_components);
X_test_pca = X_test * coeff(:, 1:num_components);

fprintf('PCA applied: %d components retained (%.2f%% variance explained)\n', ...
        num_components, explained_variance(num_components));

% Evaluate KNN for k values from 1 to 20 (step of 2)
k_values = 1:2:20;
errors = zeros(size(k_values));
accuracies = zeros(size(k_values));

best_k = 1;
best_accuracy = 0;

for i = 1:length(k_values)
    k = k_values(i);
    knn_model = fitcknn(X_train_pca, y_train, 'NumNeighbors', k);
    y_pred = predict(knn_model, X_test_pca);
    
    % Compute error rate and accuracy
    error_rate = sum(y_pred ~= y_test) / length(y_test);
    accuracy = 100 - (error_rate * 100);
    
    fprintf('k=%d → Accuracy: %.2f%% | Error: %.2f%%\n', k, accuracy, error_rate * 100);
    
    errors(i) = error_rate * 100;
    accuracies(i) = accuracy;

    % Store the best model based on accuracy
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_k = k;
        best_model = knn_model;
    end
end

fprintf('Best k found: %d with %.2f%% accuracy\n', best_k, best_accuracy);

% Generate the Elbow Method and Accuracy graphs
figure;

% Elbow Method plot (Error vs k)
subplot(1,2,1);
plot(k_values, errors, '-o', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'b');
title('Elbow Method - Error vs k');
xlabel('Number of Neighbors (k)');
ylabel('Error (%)');
grid on;
hold on;
plot(best_k, errors(k_values == best_k), 'ro', 'MarkerSize', 8, 'MarkerFaceColor', 'r');
text(best_k, errors(k_values == best_k), sprintf('  k=%d (Error: %.2f%%)', best_k, errors(k_values == best_k)), 'FontSize', 10, 'FontWeight', 'bold');
hold off;

% Accuracy vs k plot
subplot(1,2,2);
plot(k_values, accuracies, '-s', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'g');
title('KNN Model Accuracy');
xlabel('Number of Neighbors (k)');
ylabel('Accuracy (%)');
grid on;
hold on;
plot(best_k, accuracies(k_values == best_k), 'mo', 'MarkerSize', 8, 'MarkerFaceColor', 'm');
text(best_k, accuracies(k_values == best_k), sprintf('  k=%d (Accuracy: %.2f%%)', best_k, accuracies(k_values == best_k)), 'FontSize', 10, 'FontWeight', 'bold');
hold off;

% Ensure 'results' directory exists
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Save the best trained model and PCA coefficients
save(fullfile(results_folder, 'model_knn.mat'), 'best_model');
save(fullfile(results_folder, 'pca_coeff.mat'), 'coeff', 'num_components');

fprintf('KNN model and PCA coefficients saved in results/\n');
