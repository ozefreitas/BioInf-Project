# para dl_config sem remover pontuação!!!! ou a remover tags de html
# para stopswords gerais

from web_pubmed_reader import pmids_to_docs
from dl_config import DLConfig
from wrappers.pandas_wrapper import docs_to_pandasdocs, docs_to_pandasdocs_idtitabs, pandas_column_aslist, \
    relevances_to_pandas
import numpy as np
import pandas as pd
import string
from stop_words import get_stop_words
import sys
import os
import tensorflow
from dl import DL_preprocessing
from dl_models import HAN_opt
from embeddings import compute_embedding_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dl import plot_training_history
from dl import average_precision  # vai dar uma prob para um dado doc ser relevant ou nao
from tensorflow.keras.preprocessing import text
from dl import plot_roc_n_pr_curves
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import ConfigProto
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.compat.v1.keras.backend as K
import wget

seed_value = 11111
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
K.set_session(session)

print(tensorflow.__version__)

print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))

multiple_gpus = [0]  # ?????????
if multiple_gpus:
    devices = []
    for gpu in multiple_gpus:
        devices.append('/gpu:' + str(gpu))
    strategy = tensorflow.distribute.MirroredStrategy(devices=devices)

sys.path.append("../")
sys.path

from nltk.corpus import stopwords
stops = set(stopwords.words("english"))

dl_config = DLConfig(remove_punctuation="HTML", stop_words=stops, lower=True, split_by_hyphen=True)


df = pd.read_excel("C:/Users/Zé Freitas/Desktop/Mestrado/2ºSemestre/Projeto/code/datasets/dataset_final.xlsx",
                   index_col=0)
idsfinal = pandas_column_aslist(df, "Document")
docsfinal = pmids_to_docs(idsfinal, "pg42872@alunos.uminho.pt", dl_config)
docsfinal = docsfinal[0]
docsfinal[0].abstract_string
dataset_docs = docs_to_pandasdocs(docsfinal)
dataset_docs.to_csv("C:/Users/Zé Freitas/Desktop/Mestrado/2ºSemestre/Projeto/code/datasets/dataset_teste_sempunct.csv")
datasetteste = docs_to_pandasdocs_idtitabs(docsfinal, "ninteressa")
datasetteste.to_excel("C:/Users/Zé Freitas/Desktop/Mestrado/2ºSemestre/Projeto/code/datasets/dataset_teste_sempunct.xlsx")

x_total = dataset_docs
y_total = pd.read_excel("C:/Users/Zé Freitas/Desktop/Mestrado/2ºSemestre/Projeto/code/datasets/dataset_final.xlsx",
                        index_col=1, usecols="B:F")

print(x_total)

relevance = pandas_column_aslist(y_total, "Relevance")
y_total = relevances_to_pandas(y_total, relevance)
print(y_total)

print(x_total["Document"][0].title_string)
print(x_total["Document"][0].abstract_string)

X_train_df, X_test_df, y_train_df, y_test_df = train_test_split(x_total, y_total, test_size=0.3, random_state=42,
                                                                stratify=y_total)
print(X_train_df.shape)
print(X_test_df.shape)
print(y_train_df.shape)
print(y_test_df.shape)
y_test_df.values

model_name = "HC_pro"
dl_config = DLConfig(model_name=model_name, seed_value=seed_value)
dl_config.stop_words = stops
dl_config.lower = True
dl_config.remove_punctuation = False
dl_config.split_by_hyphen = True
dl_config.lemmatization = False
dl_config.stems = False

#Parameters
dl_config.padding = 'post'            #'pre' -> default; 'post' -> alternative
dl_config.truncating = 'post'         #'pre' -> default; 'post' -> alternative      #####
dl_config.oov_token = 'OOV'

dl_config.epochs = 50
dl_config.batch_size = 32     # e aumentar o batch
dl_config.learning_rate = 0.001   #experimentar diminuir

dl_config.max_sent_len = 50      #sentences will have a maximum of "max_sent_len" words    #400/500
dl_config.max_nb_words = 100_000      #it will only be considered the top "max_nb_words" words in the dataset
dl_config.max_nb_sentences = 15    # set only for the hierarchical attention model!!!
# no novo usar um maior numero de palavras
dl_config.embeddings = 'biowordvec'

if not os.path.isdir('./embeddings'):
    os.mkdir('./embeddings')

if dl_config.embeddings == 'biowordvec':   #200 dimensions
    if not os.path.isfile('./embeddings/biowordvec'):
        url = "https://ndownloader.figshare.com/files/12551780"
        output_directory = "./embeddings/biowordvec"
        wget.download(url, out=output_directory)
    #    !wget -O ./embeddings/biowordvec https://ndownloader.figshare.com/files/12551780
    dl_config.embedding_path = './embeddings/biowordvec'
    dl_config.embedding_dim = 200
    dl_config.embedding_format = 'word2vec'

dl_config.keras_callbacks = True

 # compara as losses d otrain e validation
if dl_config.keras_callbacks:
    dl_config.patience = 5   #early-stopping patience
    checkpoint_path = str(dl_config.model_id_path) + '\checkpoint.hdf5'
    keras_callbacks = [
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=dl_config.patience),
           ModelCheckpoint(checkpoint_path, monitor='val_loss', mode='min', verbose=1, save_best_only=True)  # guarda o modelo que tem a menor loss no validation
    ]
else:
    keras_callbacks = None

dl_config.tokenizer = text.Tokenizer(num_words=dl_config.max_nb_words, oov_token=dl_config.oov_token)

dl_config.validation_percentage = 30  # talvez aumentar
X_train, y_train, x_val, y_val = DL_preprocessing(X_train_df, y_train_df,
                                                  dl_config, dataset='train',
                                                  validation_percentage = dl_config.validation_percentage,
                                                  seed_value=dl_config.seed_value)
X_train[0][0]

dl_config.embedding_matrix = compute_embedding_matrix(dl_config, embeddings_format = dl_config.embedding_format)
# quantas palavras encontrou no vocabulario do embedding

print(dl_config.embedding_matrix.shape)
print(dl_config.embedding_matrix)


if multiple_gpus:
    with strategy.scope():
        model = HAN_opt(dl_config.embedding_matrix, dl_config, learning_rate=dl_config.learning_rate,
                                               seed_value=dl_config.seed_value)
else:
    model = HAN_opt(dl_config.embedding_matrix, dl_config, learning_rate=dl_config.learning_rate,
                                                seed_value=dl_config.seed_value)


history = model.fit(X_train, y_train,
                    epochs=dl_config.epochs,
                    batch_size=dl_config.batch_size,
                    validation_data=(x_val,y_val),
                    callbacks=keras_callbacks)

if dl_config.keras_callbacks:
    model.load_weights(checkpoint_path)

train_loss, dl_config.train_acc = model.evaluate(X_train, y_train, verbose=0, batch_size = dl_config.batch_size)
plot_training_history(history_dict=history, dl_config=dl_config)

print('Training Loss: %.3f' % (train_loss))
print('Training Accuracy: %.3f' % (dl_config.train_acc))
# loss do training e validation, se o loss do validation for muito elevado nao é bom, quer se sempre diminuir o maximo
# possivel o loss

# testar com os 606
# e ter cuidado com o numero de negativos no test

x_test, y_test = DL_preprocessing(X_test_df, y_test_df, dl_config, dataset='test')

yhat_probs = model.predict(x_test, verbose=0)
yhat_probs = yhat_probs[:, 0]

yhat_classes = np.where(yhat_probs > 0.5, 1, yhat_probs)
yhat_classes = np.where(yhat_classes < 0.5, 0, yhat_classes).astype(np.int64)

dl_config.test_roc_auc, dl_config.test_pr_auc = plot_roc_n_pr_curves(y_test, yhat_probs, dl_config=dl_config)

# ROC AUC
print('ROC AUC: %f' % dl_config.test_roc_auc)

# avg precision
dl_config.test_avg_prec = average_precision(y_test_df, yhat_probs)
print('Average Precision: %f' % dl_config.test_avg_prec)

# accuracy
dl_config.test_acc = accuracy_score(y_test, yhat_classes)
print('Accuracy: %f' % dl_config.test_acc)

# precision tp / (tp + fp)
dl_config.test_prec = precision_score(y_test, yhat_classes)
print('Precision: %f' % dl_config.test_prec)

# recall: tp / (tp + fn)
dl_config.test_recall = recall_score(y_test, yhat_classes)
print('Recall: %f' % dl_config.test_recall)

# f1: 2 tp / (2 tp + fp + fn)
dl_config.test_f1_score = f1_score(y_test, yhat_classes)
print('F1 score: %f' % dl_config.test_f1_score)

# confusion matrix
matrix = confusion_matrix(y_test, yhat_classes)
print('Confusion Matrix:\n %s \n' % matrix)
