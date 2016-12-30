''' This function print the main results for our method '''
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

### path to ground truth labels
labels_path = './y_true.txt'

### path to class predictions file
class_predictions_path = './y_pred.txt'
### path to predictions file
predictions_score_path = './y_score.txt'

### Read files
y_true = np.loadtxt(labels_path)
y_pred = np.loadtxt(class_predictions_path)
y_score = np.loadtxt(predictions_score_path)

### Saving Paths
model_name = "segmented_skin_classifier"
#model_path = "models_trained/" +model_name+"/"

f_eval = open("metrics.txt", 'w')
f_eval.write('Model evaluation of ' + str(model_name) + '\n' + str('-'*30)+ '\n' )

# Compute TP, FP, TN, FN
# (since my file has Benign=1, Malignant=0, P=0, N=1)
TP = np.sum(np.logical_and(y_pred == 0, y_true == 0))
TN = np.sum(np.logical_and(y_pred == 1, y_true == 1))
FP = np.sum(np.logical_and(y_pred == 0, y_true == 1))
FN = np.sum(np.logical_and(y_pred == 1, y_true == 0))

print('-'*30)
print('False Negatives: ', str(FN))
print('-'*30)
print('True Positives: ', str(TP))
print('-'*30)
print('False Positives: ', str(FP))
print('-'*30)
print('True Negatives: ', str(TN))
print('-'*30)

# Compute Accuracy classification score.
acc = metrics.accuracy_score(y_true, y_pred)
f_eval.write('Accuracy: ' + str(acc) + '\n')
print('Accuracy: ', str(acc))
print('-'*30)

# Compute the Sensitivity (recall)
sensitivity = TP/float(TP + FN)
f_eval.write('Sensitivity (recall) score: ' + str(sensitivity) + '\n')
print('Sensitivity (recall) score: ', str(sensitivity))
print('-'*30)

# Compute Specificity score.
specificity = TN/float(TN+FP)
f_eval.write('Specificity: ' + str(specificity) + '\n')
print('Specificity: ', str(specificity))
print('-'*30)

# Compute the precision
precision = metrics.precision_score(y_true, y_pred)
f_eval.write('Precision score: ' + str(precision) + '\n')
print('Precision score: ', str(precision))
print('-'*30)

# Compute the F1 score, also known as balanced F-score or F-measure
f1 = metrics.f1_score(y_true, y_pred)
f_eval.write('F1 score: ' + str(f1) + '\n')
print('F1 score: ', str(f1))
print('-'*30)

# Compute the F-beta score
f_beta = metrics.fbeta_score(y_true, y_pred,beta=0.5)
f_eval.write('F-beta (b=0.5) score: ' + str(f_beta) + '\n')
print('F-beta (b=0.5) score: ', str(f_beta))
print('-'*30)

# Compute the average Hamming loss.
hamming = metrics.hamming_loss(y_true, y_pred)
f_eval.write('Hamming loss: ' + str(hamming) + '\n')
print('Hamming loss: ', str(hamming))
print('-'*30)

# Compute Jaccard similarity coefficient score
jacc = metrics.jaccard_similarity_score(y_true, y_pred)
f_eval.write('Jaccard similarity coefficient score: ' + str(jacc) + '\n')
print('Jaccard similarity coefficient score: ', str(jacc))
print('-'*30)

# Compute the Matthews correlation coefficient (MCC) for binary classes
matthews = metrics.matthews_corrcoef(y_true, y_pred)
f_eval.write('Matthews correlation coefficient (MCC): ' + str(matthews) + '\n')
print('Matthews correlation coefficient (MCC): ', str(matthews))
print('-'*30)

# Compute Mean absolute error regression loss
mae_loss = metrics.mean_absolute_error(y_true, y_pred)
f_eval.write('Mean absolute error regression loss: ' + str(mae_loss) + '\n')
print('Mean absolute error regression loss: ', str(mae_loss))
print('-'*30)

# Compute the Mean squared error regression loss
msq_loss = metrics.mean_squared_error(y_true, y_pred)
f_eval.write('Mean squared error regression loss: ' + str(msq_loss) + '\n')
print('Mean squared error regression loss: ', str(msq_loss))
print('-'*30)

# Compute Area Under the Curve (AUC) from prediction scores
AUC = metrics.roc_auc_score(y_true, y_score)
f_eval.write('Area Under the Curve (AUC): ' + str(AUC) + '\n')
print('Area Under the Curve (AUC): ', str(AUC))
print('-'*30)

# Compute Receiver operating characteristic (ROC)
ROC = metrics.roc_curve(y_true, y_score)
f_eval.write('Receiver operating characteristic (ROC): ' + str(ROC) + '\n')
print('Receiver operating characteristic (ROC): ', str(ROC))
print('-'*30)

# Compute confusion matrix to evaluate the accuracy of a classification
confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
f_eval.write('Confusion matrix: ' + str(confusion_matrix) + '\n')
print('Confusion matrix: ', str(confusion_matrix))
print('-'*30)

norm_conf = []
for i in confusion_matrix:
        a = 0
        tmp_arr = []
        a = sum(i,0)
        for j in i:
                tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

fig = plt.figure()
ax = fig.add_subplot(111)
res = ax.imshow(norm_conf, interpolation='nearest')
for i, cas in enumerate(confusion_matrix):
    for j, c in enumerate(cas):
        if c>0:
            plt.text(j-.2, i+.2, c, color="white",fontsize=14)
cb = fig.colorbar(res)
plt.title('Segmented Skin Lesion Classifier')
#ax.set_xticklabels(['']+labels, fontsize=13)
#ax.set_yticklabels(['']+labels,fontsize=13)
plt.xlabel('Predicted',fontsize=16)
plt.ylabel('True',fontsize=16)
plt.savefig("confmat.png", format="png",dpi=400)

f_eval.close()
