clc; clear; close all;

% Load dataset with extracted features
load(fullfile('src', 'knn_dataset.mat'));

% Normalize data using Z-score (same as KNN)
X = zscore(X);

% Split data: 80% training, 20% testing
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Load PCA coefficients from KNN preprocessing
load(fullfile('results', 'pca_coeff.mat'));  
X_train_pca = X_train * coeff(:, 1:num_components);
X_test_pca = X_test * coeff(:, 1:num_components);

fprintf('PCA applied: %d components retained for SVM\n', num_components);

% Evaluate different kernels to determine the best one
kernels = {'linear', 'rbf', 'polynomial'};
accuracies = zeros(size(kernels));

best_accuracy = 0;
best_kernel = '';

for i = 1:length(kernels)
    kernel = kernels{i};
    fprintf('Testing kernel: %s\n', kernel);
    
    template = templateSVM('KernelFunction', kernel, 'BoxConstraint', 1.0);
    svm_model = fitcecoc(X_train_pca, y_train, 'Learners', template, 'KFold', 5);
    
    y_pred = kfoldPredict(svm_model);
    accuracy = sum(y_pred == y_train) / length(y_train) * 100;
    
    fprintf('Kernel %s → Accuracy: %.2f%%\n', kernel, accuracy);
    accuracies(i) = accuracy;
    
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_kernel = kernel;
        best_model = svm_model;
    end
end

% Display best kernel found
fprintf('Best kernel: %s with %.2f%% accuracy\n', best_kernel, best_accuracy);

% Plot Kernel Accuracy Comparison
figure;
bar(categorical(kernels), accuracies, 'FaceColor', 'b');
title('Kernel Accuracy Comparison');
xlabel('Kernel Type');
ylabel('Accuracy (%)');
grid on;
hold on;
[max_acc, idx] = max(accuracies);
text(idx, max_acc, sprintf('%.2f%%', max_acc), 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
hold off;

% Ensure 'results' directory exists
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Save best trained SVM model
save(fullfile(results_folder, 'model_svm.mat'), 'best_model');
fprintf('SVM model saved in results/model_svm.mat\n');

% Evaluate best SVM model on test set
model_svm = best_model.Trained{1};  
y_pred = predict(model_svm, X_test_pca);
