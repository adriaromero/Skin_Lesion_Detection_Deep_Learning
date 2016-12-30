function [ ] = plot_scores( )
%ACC_PLOT plots Loss and Accuracy vs Epoch 

train_acc = csvread('train_acc.txt');
train_loss = csvread('train_loss.txt');
train_recall = csvread('train_recall.txt');
train_precision = csvread('train_precision.txt');

val_acc = csvread('val_acc.txt');
val_loss = csvread('val_loss.txt');
val_recall = csvread('val_recall.txt');
val_precision = csvread('val_precision.txt');

y_true = csvread('y_true.txt');
y_score = csvread('y_score.txt');

epochs = length(train_acc);

% Plot Precision-Recall curve
prec_rec(y_score, y_true);

%Plot accuracy
x = linspace(0,epochs,epochs);
figure
plot(x,train_acc,'DisplayName','Training')
hold on
plot(x,val_acc, 'DisplayName','Validation')

title('Accuracy')
xlabel('Epochs')
ylabel('Normalized Accuracy')
legend('show')

%Plot Loss
figure
plot(x,train_loss,'DisplayName','Training')
hold on
plot(x,val_loss, 'DisplayName','Validation')

title('Loss')
xlabel('Epochs')
ylabel('Loss (Binary Cross Entropy)')
legend('show')

%Plot Recall
figure
plot(x,train_recall,'DisplayName','Training')
hold on
plot(x,fliplr(val_recall), 'DisplayName','Validation')

title('Recall')
xlabel('Epochs')
ylabel('Recall value')
legend('show')

%Plot Precision
figure
plot(x,train_precision,'DisplayName','Training')
hold on
plot(x,val_precision, 'DisplayName','Validation')

title('Precision')
xlabel('Epochs')
ylabel('Precision value')
legend('show')

% Compute best model metrics
best_acc = max(val_acc);
best_loss = min(val_loss);
best_recall = max(val_recall);
best_precision = max(val_precision);

fprintf('Best Validation Accuracy: %f \n', best_acc);
fprintf('Best Validation Loss: %f \n', best_loss);
fprintf('Best Validation Recall: %f \n', best_recall);
fprintf('Best Validation Precision: %f \n', best_precision);

end
