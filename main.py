import re
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from tqdm.auto import tqdm
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import matplotlib.pyplot as plt
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from tokenizers import AddedToken
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import os
import json
from A.model import Paragraph_Preprocess,Paragraph_Eng,Sentence_Eng, qwk_obj,Word_Preprocess,Word_Eng, quadratic_weighted_kappa,Sentence_Preprocess
from B.model import preprocess_and_tokenize,preprocess_text,compute_metrics,prepare_datasets
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.simplefilter('ignore')
# ======================================================================================================================
# Load Data
columns = [  
    (
        pl.col("full_text").str.split(by="\n\n").alias("paragraph")
    ),
]
PATH = './Datasets/'

train = pl.read_csv(PATH + "train.csv").with_columns(columns)
test = pl.read_csv(PATH + "test.csv").with_columns(columns)

# ======================================================================================================================
# Task A
tmp = Paragraph_Preprocess(train)
train_feats = Paragraph_Eng(tmp)
train_feats['score'] = train['score']
feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
tmp = Sentence_Preprocess(train)
train_feats = train_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')
feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
tmp = Word_Preprocess(train)
train_feats = train_feats.merge(Word_Eng(tmp), on='essay_id', how='left')
feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))
# TfidfVectorizer parameter
vectorizer = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            token_pattern=None,
            strip_accents='unicode',
            analyzer = 'word',
            ngram_range=(1,3),
            min_df=0.05,
            max_df=0.95,
            sublinear_tf=True,
)

train_tfid = vectorizer.fit_transform([i for i in train['full_text']])
dense_matrix = train_tfid.toarray()
df = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
df.columns = tfid_columns
df['essay_id'] = train_feats['essay_id']
train_feats = train_feats.merge(df, on='essay_id', how='left')
feature_names = list(filter(lambda x: x not in ['essay_id','score'], train_feats.columns))

history = {
    'train_loss': [],
    'val_loss': [],
    'qwk_score': []
}
LOAD = False
models = []
a = 2.948
b = 1.092
if LOAD:
    for i in range(5):
        models.append(lgb.Booster(model_file=f'Results/fold_{i}.txt'))
else:
    # OOF is used to store the prediction results of each model on the validation set
    oof = []
    x= train_feats
    y= train_feats['score'].values
    # 5 fold
    kfold = KFold(n_splits=5, random_state=42, shuffle=True)
    callbacks = [log_evaluation(period=25), early_stopping(stopping_rounds=75,first_metric_only=True)]
    history['train_qwk'] = []
    history['val_qwk'] = []
    for fold_id, (trn_idx, val_idx) in tqdm(enumerate(kfold.split(x.copy(), y.copy().astype(str)))):
        model = lgb.LGBMRegressor(
                objective = qwk_obj,
                metrics = 'None',
                learning_rate = 0.1,
                max_depth = 5,
                num_leaves = 10,
                colsample_bytree=0.5,
                reg_alpha = 0.1,
                reg_lambda = 0.8,
                n_estimators=1024,
                random_state=42,
                verbosity = - 1)
        
        X_train = train_feats.iloc[trn_idx][feature_names]
        Y_train = train_feats.iloc[trn_idx]['score'] - a

        X_val = train_feats.iloc[val_idx][feature_names]
        Y_val = train_feats.iloc[val_idx]['score'] - a

        print('\nFold_{} Training ================================\n'.format(fold_id+1))
        # Training model
        lgb_model = model.fit(X_train,Y_train,
                                eval_names=['train', 'valid'],
                                eval_set=[(X_train, Y_train), (X_val, Y_val)],
                                eval_metric=quadratic_weighted_kappa,
                                callbacks=callbacks,)
        history['train_qwk'].append(model.best_score_['train']['QWK'])
        history['val_qwk'].append(model.best_score_['valid']['QWK'])

    
        # Use the trained model to predict the validation set
        pred_val = lgb_model.predict(
            X_val, num_iteration=lgb_model.best_iteration_)
        df_tmp = train_feats.iloc[val_idx][['essay_id', 'score']].copy()
        df_tmp['pred'] = pred_val + a
        oof.append(df_tmp)
        # Save model parameters
        models.append(model.booster_)
        lgb_model.booster_.save_model(f'./A/fold_{fold_id}.txt')
    df_oof = pd.concat(oof)

# Plot the learning curve of the QWK score
plt.figure(figsize=(12, 6))
plt.plot(history['train_qwk'], label='Train QWK')
plt.title('QWK Score Over Folds')
plt.xlabel('Fold')
plt.ylabel('QWK Score')
plt.legend()
plt.savefig("./A/QWK_TaskA.png")
plt.show()

X_test = train_feats[feature_names]
Y_test = train_feats['score'].astype(int)

# Make predictions using the best model
pred_test = np.mean([model.predict(X_test, num_iteration=model.best_iteration) for model in models], axis=0)+a
pred_test_rounded = np.round(pred_test).clip(1, 6)  

# Calculate the confusion matrix
cm = confusion_matrix(Y_test, pred_test_rounded)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Test Data')
plt.savefig("./A/CM_TaskA.png")
plt.show()

# Paragraph
tmp = Paragraph_Preprocess(test) 
test_feats = Paragraph_Eng(tmp)
# Sentence
tmp = Sentence_Preprocess(test)
test_feats = test_feats.merge(Sentence_Eng(tmp), on='essay_id', how='left')
# Word
tmp = Word_Preprocess(test)
test_feats = test_feats.merge(Word_Eng(tmp), on='essay_id', how='left')
# Tfidf
test_tfid = vectorizer.transform([i for i in test['full_text']])
dense_matrix = test_tfid.toarray()
df = pd.DataFrame(dense_matrix)
tfid_columns = [ f'tfid_{i}' for i in range(len(df.columns))]
df.columns = tfid_columns
df['essay_id'] = test_feats['essay_id']
test_feats = test_feats.merge(df, on='essay_id', how='left')
# Features number
feature_names = list(filter(lambda x: x not in ['essay_id','score'], test_feats.columns))
print('Features number: ',len(feature_names))
print(test_feats.head(3))

prediction = test_feats[['essay_id']].copy()
prediction['score'] = 0
pred_test = models[0].predict(test_feats[feature_names]) + a
for i in range(4):
    pred_now = models[i+1].predict(test_feats[feature_names]) + a
    pred_test = np.add(pred_test,pred_now)

pred_test = pred_test/5
print(pred_test)

pred_test = pred_test.clip(1, 6).round()
prediction['score'] = pred_test
prediction.to_csv('A/submission_A.csv', index=False)
print(prediction.head(3))

# ======================================================================================================================
# Task B
Retrain = False

initial_model = './B/DeBERTA_initial'
# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(initial_model)
model = AutoModelForSequenceClassification.from_pretrained(initial_model)
# Load Data
data = pd.read_csv('./Datasets/train.csv')[:100]
data['label'] = data['score'].apply(lambda x: x-1).astype('float32')
class CFG:
    n_splits = 5
    seed = 42
    max_length = 1024
    lr = 1e-5
    train_batch_size = 4
    eval_batch_size = 8
    train_epochs = 4
    weight_decay = 0.01
    warmup_ratio = 0.0
    num_labels = 6 
skf = StratifiedKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)
for i, (_, val_index) in enumerate(skf.split(data, data["score"])):
    data.loc[val_index, "fold"] = i

tokenizer = AutoTokenizer.from_pretrained(initial_model)
tokenizer.add_tokens(["\n", "  "])  
tokenizer.add_tokens([AddedToken(" "*2, normalized=False)])

data['full_text'] = data['full_text'].apply(preprocess_text)

tokenizer = AutoTokenizer.from_pretrained(initial_model)
tokenizer.add_tokens(['[NL]', '[DS]'])

if Retrain:
    for fold in range(CFG.n_splits):
        output_dir = f'./B/DeBERTa_{fold}'
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy='epoch',  
            save_strategy='epoch',       
            save_total_limit=1,
            learning_rate=CFG.lr,
            per_device_train_batch_size=CFG.train_batch_size,
            per_device_eval_batch_size=CFG.eval_batch_size,
            num_train_epochs=CFG.train_epochs,
            weight_decay=CFG.weight_decay,
            load_best_model_at_end=True, 
            metric_for_best_model='qwk',  
            greater_is_better=True,       
            fp16=False
        )

        train_dataset = prepare_datasets(fold)
        eval_dataset = prepare_datasets(fold)

        model = AutoModelForSequenceClassification.from_pretrained(initial_model, num_labels=CFG.num_labels, ignore_mismatched_sizes=True)
        model.resize_token_embeddings(len(tokenizer))

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )
        trainer.train()

        # Save Best model
        model.save_pretrained(f'./B/best_model')

    
test_data = pd.read_csv('./Datasets/test.csv')

test_dataset = preprocess_and_tokenize(test_data)
from transformers import Trainer

# Load model
model = AutoModelForSequenceClassification.from_pretrained('./B/best_model')
model.resize_token_embeddings(len(tokenizer)) 
trainer = Trainer(model=model)

log_file_path = './B/DeBERTa_0/checkpoint-80/trainer_state.json'

with open(log_file_path, 'r') as file:
        json_data = json.load(file)

# Extracting log history which contains the metrics per epoch
log_history = json_data['log_history']

# Initialize lists to store the epochs, evaluation loss, and QWK scores
epochs = []
eval_losses = []
eval_qwks = []

# Populate the lists with data from the log history
for entry in log_history:
    epochs.append(entry['epoch'])
    eval_losses.append(entry['eval_loss'])
    eval_qwks.append(entry['eval_qwk'])

# Creating plots for the QWK score
plt.figure(figsize=(12, 6))
plt.plot(epochs, eval_qwks, label='Eval QWK')
plt.title('QWK Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('QWK Score')
plt.legend()
plt.savefig("./B/QWK_TaskB.png")
plt.show()

# Make predication
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)
predicted_labels += 1

# Save our prediction as submission
submission = pd.read_csv('./Datasets/sample_submission.csv')
submission['label'] = predicted_labels
submission.to_csv('./B/submission_B.csv', index=False)
print('Submission of Task B saved.')

# Use training data to create a confusion matrix
trainer = Trainer(model=model)

train_data = pd.read_csv('Datasets/train.csv')
train_data['full_text'] = train_data['full_text'].apply(preprocess_text)  
train_dataset = preprocess_and_tokenize(train_data)  

predictions = trainer.predict(train_dataset)
predicted_labels = predictions.predictions.argmax(axis=-1)
predicted_labels += 1

actual_labels = train_data['score']  
cm = confusion_matrix(actual_labels, predicted_labels)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig("./Results/CM_TaskB.png")
plt.show()



