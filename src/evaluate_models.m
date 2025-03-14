clc; clear; close all;

% Load the test dataset
load(fullfile('src', 'knn_dataset.mat'));

% Split data: 80% training, 20% testing
cv = cvpartition(size(X,1), 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train = y(training(cv), :);
X_test = X(test(cv), :);
y_test = y(test(cv), :);

% Load trained models
if exist(fullfile('results', 'model_knn.mat'), 'file')
    load(fullfile('results', 'model_knn.mat'));
    model_knn = best_model;
else
    error('Model model_knn.mat not found in results/. Train KNN before evaluation.');
end

if exist(fullfile('results', 'model_svm.mat'), 'file')
    load(fullfile('results', 'model_svm.mat'));
    model_svm = best_model.Trained{1};
else
    error('Model model_svm.mat not found in results/. Train SVM before evaluation.');
end

% Normalize test data and apply PCA
X_test = zscore(X_test);
load(fullfile('results', 'pca_coeff.mat'));
X_test_pca = X_test * coeff(:, 1:num_components);
fprintf('X_test adjusted to %d features using PCA.\n', size(X_test_pca,2));

% Predictions with KNN and SVM
y_pred_knn = predict(model_knn, X_test_pca);
y_pred_svm = predict(model_svm, X_test_pca);

% Compute evaluation metrics
metrics_knn = compute_metrics(y_test, y_pred_knn);
metrics_svm = compute_metrics(y_test, y_pred_svm);

% Display evaluation metrics
fprintf('\nKNN Model Evaluation:\n');
disp(metrics_knn);

fprintf('\nSVM Model Evaluation:\n');
disp(metrics_svm);

% Plot confusion matrices
figure;
subplot(1,2,1);
confusionchart(y_test, y_pred_knn);
title('Confusion Matrix - KNN');

subplot(1,2,2);
confusionchart(y_test, y_pred_svm);
title('Confusion Matrix - SVM');

% ROC and AUC Curve
figure;
hold on;
plot_roc(y_test, y_pred_knn, 'KNN', 'b');
plot_roc(y_test, y_pred_svm, 'SVM', 'r');
hold off;
legend('KNN', 'SVM');
title('ROC and AUC Curve');
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
grid on;

% F1-Score per class comparison
figure;
bar([metrics_knn.F1_per_class(:), metrics_svm.F1_per_class(:)], 'grouped');
legend('KNN', 'SVM', 'Location', 'northwest');
xticklabels({'Fist', 'Open', 'Pinch', 'Relax', 'WaveIn', 'WaveOut'});
title('F1-Score Comparison by Class');
xlabel('Gestures');
ylabel('F1-Score (%)');
grid on;

% Function to compute evaluation metrics
function metrics = compute_metrics(y_true, y_pred)
    confMat = confusionmat(y_true, y_pred);
    TP = diag(confMat);
    FP = sum(confMat, 1)' - TP;
    FN = sum(confMat, 2) - TP;
    TN = sum(confMat(:)) - (TP + FP + FN);

    precision = TP ./ (TP + FP);
    recall = TP ./ (TP + FN);
    f1_score = 2 * (precision .* recall) ./ (precision + recall);
    accuracy = sum(TP) / sum(confMat(:));

    precision(isnan(precision)) = 0;
    recall(isnan(recall)) = 0;
    f1_score(isnan(f1_score)) = 0;

    metrics.Precision = mean(precision) * 100;
    metrics.Recall = mean(recall) * 100;
    metrics.F1_Score = mean(f1_score) * 100;
    metrics.Accuracy = accuracy * 100;
    metrics.F1_per_class = f1_score * 100;
    metrics.Precision_per_class = precision * 100;
    metrics.Recall_per_class = recall * 100;
end

% Function to plot ROC curve
function plot_roc(y_true, y_pred, model_name, color)
    [X, Y, ~, AUC] = perfcurve(y_true, y_pred, 1);
    plot(X, Y, 'Color', color, 'LineWidth', 2);
    text(0.6, 0.2 + 0.05 * strcmp(model_name, 'SVM'), ...
        sprintf('%s AUC = %.2f', model_name, AUC), 'Color', color, 'FontSize', 10, 'FontWeight', 'bold');
end
