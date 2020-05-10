"""
TODO update description
based on this: https://raw.githubusercontent.com/explosion/spaCy/master/examples/deep_learning_keras.py
This example shows how to use an LSTM sentiment classification model trained
using Keras in spaCy. spaCy splits the document into sentences, and each
sentence is classified using the LSTM. The scores for the sentences are then
aggregated to give the document score. This kind of hierarchical model is quite
difficult in "pure" Keras or Tensorflow, but it's very effective. The Keras
example on this dataset performs quite poorly, because it cuts off the documents
so that they're a fixed size. This hurts review accuracy a lot, because people
often summarise their rating in the final sentence

Prerequisites:
spacy download en_vectors_web_lg
pip install keras==2.0.9

Compatible with: spaCy v2.0.0+
"""

import plac
import random
import pathlib
import cytoolz
import numpy
import keras as ks
from keras.layers.wrappers import  TimeDistributed
from keras.layers  import Dense,CuDNNLSTM, GRU,Conv1D,Flatten,Conv2D,Conv3D,UpSampling3D,UpSampling2D,BatchNormalization,Activation,MaxPooling3D,Reshape,ConvLSTM2D,LSTM,MaxPooling2D
from keras.layers import Input,Dropout,GlobalMaxPooling3D,SimpleRNN,MaxPooling1D,concatenate,AveragePooling1D,GlobalAveragePooling3D,LeakyReLU,Embedding,Bidirectional,TimeDistributed
from keras.models import Model
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import SimpleRNN
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve,auc,r2_score
#from skll.metrics import pearson,spearman
from keras import backend as K
from scipy.stats import ortho_group
from scipy.stats import pearsonr,spearmanr
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.utils.training_utils import multi_gpu_model
from numpy.random import seed
from keras.optimizers import RMSprop, Adam,SGD
from spacy.compat import pickle
import spacy
import corclass
import roc_auc
import classf
from spacy.tokens import Doc
import srsly
import pandas as pd
import time
import os
from src import textcat
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import roc_auc_score

class LSTM_textcat(object):
    name = "lstm_textcat"   # update for different pipeline components

    @classmethod
    def load(cls, path, nlp, max_length=1500):
        with (path / "config.json").open() as file_:
            model = model_from_json(file_.read())
        with (path / "model").open("rb") as file_:
            lstm_weights = pickle.load(file_)
        embeddings = get_embeddings(nlp.vocab)
        model.set_weights([embeddings] + lstm_weights)
        return cls(model, max_length=max_length)

    # TODO not tested
    def save(self, model_dir):
        weights = self._model.get_weights()
        if model_dir is not None:
            with (model_dir / "model").open("wb") as file_:
                pickle.dump(weights[1:], file_)
            with (model_dir / "config.json").open("w") as file_:
                file_.write(self._model.to_json())

    def __init__(self, model, cats, max_length=1500):
        self._model = model
        self.max_length = max_length
        self.cats = cats

    def __call__(self, doc):
        X = get_features([doc], self.max_length)
        y = self._model.predict(X)
        self.set_categories(doc, y)
        return doc

    def pipe(self, docs, batch_size=1000):
        for minibatch in cytoolz.partition_all(batch_size, docs):
            minibatch = list(minibatch)

            # doing this per doc for now
            Xs = get_features(minibatch, self.max_length)
            ys = self._model.predict(Xs)
            for doc, y in zip(minibatch, ys):
                doc.user_data['my_cats'] = dict(zip(self.cats, y))

            for doc in minibatch:
                yield doc

            # # TODO sentence-level classification and final aggregation
            # sentences = []
            # for doc in minibatch:
            #     sentences.extend(doc.sents)
            # Xs = get_features(sentences, self.max_length)
            # ys = self._model.predict(Xs)
            # for sent, label in zip(sentences, ys):
            #     #sent.doc.sentiment += label - 0.5
            #     # TODO fix this
            #     sent.doc.categories += label - 0.5
            # for doc in minibatch:
            #     yield doc

    def set_categories(self, doc, y):
        # TODO use doc extensions instead of user data
        # Doc.set_extension("hello", default=True)
        # doc._.hello = False
        doc.user_data['my_cats'] = dict(zip(self.cats, y))


# TODO sentence-level classifier
def get_labelled_sentences(docs, doc_labels):
    labels = []
    sentences = []
    for doc, y in zip(docs, doc_labels):
        for sent in doc.sents:
            sentences.append(sent)
            labels.append(y)
    return sentences, numpy.asarray(labels, dtype="int32")


def get_features(docs, max_length):
    docs = list(docs)
    Xs = numpy.zeros((len(docs), max_length), dtype="int32")
    for i, doc in enumerate(docs):
        j = 0
        for token in doc:
            vector_id = token.vocab.vectors.find(key=token.orth)
            if vector_id >= 0:
                Xs[i, j] = vector_id
            else:
                Xs[i, j] = 0
            j += 1
            if j >= max_length:
                break
    return Xs

# TODO
# @plac.annotations(
#     train_dir=("Location of training file or directory"),
#     dev_dir=("Location of development file or directory"),
#     model_dir=("Location of output model directory",),
#     is_runtime=("Demonstrate run-time usage", "flag", "r", bool),
#     nr_hidden=("Number of hidden units", "option", "H", int),
#     max_length=("Maximum sentence length", "option", "L", int),
#     dropout=("Dropout", "option", "d", float),
#     learn_rate=("Learn rate", "option", "e", float),
#     nb_epoch=("Number of training epochs", "option", "i", int),
#     batch_size=("Size of minibatches for training LSTM", "option", "b", int),
#     nr_examples=("Limit to N templates", "option", "n", int),
# )

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes

    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss
def train(
        train_paths,
        dev_paths,
        nr_hidden=64,       # network shape
        max_length=1500,
        dropout=0.5,        # optimizer params
        learn_rate=0.0001,
        nb_epoch=5,
        batch_size=256,
        n_texts=10000,
        return_keras_model=True,
        #   by_sentence=False,     # TODO implement sentence-wise models
):

    saved_args = locals()
    print("~~~~textcat args~~~~")
    # this crashes pdb but works when not debugging
    #print(srsly.json_dumps(saved_args, indent=2))
    print("~~~~~~~~~~~~~~~~~~~~")

    print("Loading spaCy")
    nlp = spacy.load("/mnt/DataA/TextClassifiyer/clinical_notes2/data/word_vectors/fasttext/result_spacy/all_notes_5_12_19-default_epochs20")
    #nlp = spacy.load("en_vectors_web_lg")#spacy.load('/mnt/DataA/TextClassifiyer/clinical_notes/data/scispacy/en_core_sci_md-0.2.0/en_core_sci_md/en_core_sci_md-0.2.0')
    #spacy.load("en_vectors_web_lg")#('/mnt/DataA/TextClassifiyer/clinical_notes/data/fasttext/en_vectors_web_lg')#("en_vectors_web_lg")
    #nlp.add_pipe(nlp.create_pipe("sentencizer"))
    embeddings = get_embeddings(nlp.vocab)
    labels0=pd.read_feather(train_paths[0])
    labels1=pd.read_feather(train_paths[1])
    labels2=pd.read_feather(train_paths[2])
    labels=pd.concat([labels0,labels1,labels2])
    #print(labels["note_id"])
    train_data = labels#textcat.read_json_data(train_paths, limit=n_texts)
    if dev_paths is not None:
        dev_data = pd.read_feather(dev_paths[0])
        print("Using up to {} templates ({} training, {} validation)".format(n_texts, len(train_data), len(dev_data)))
    else:
        print("Using up to {} templates ({} training, no validation)".format(n_texts, len(train_data)))

    #classes = train_data[0]["cats"].keys() #set([cat for row in train_data for cat in row['cats']])
    #print(classes)
    #print(train_data[0]["cats"])
    # TODO remove pandas requirement
    print(train_data.keys())
    train_CAD_labels= numpy.asarray(train_data['CAD3'], dtype="int32")
    train_ACS_labels= numpy.asarray(train_data['ACS3'], dtype="int32")
    train_HF_labels= numpy.asarray(train_data['HF3'], dtype="int32")
    train_AF_labels= numpy.asarray(train_data['AF3'], dtype="int32")

    train_CAD_OH_labels=to_categorical(((train_CAD_labels+1)/1).astype(int))
    train_ACS_OH_labels=to_categorical(((train_ACS_labels+1)/1).astype(int))
    train_HF_OH_labels=to_categorical(((train_HF_labels+1)/1).astype(int))
    train_AF_OH_labels=to_categorical(((train_AF_labels+1)/1).astype(int))

    train_CAD_BIN_labels=np.clip(np.absolute(train_CAD_labels),0,1).astype(int)
    train_ACS_BIN_labels=np.clip(np.absolute(train_ACS_labels),0,1).astype(int)
    train_HF_BIN_labels=np.clip(np.absolute(train_HF_labels),0,1).astype(int)
    train_AF_BIN_labels=np.clip(np.absolute(train_AF_labels),0,1).astype(int)

    dev_CAD_labels= numpy.asarray(dev_data['CAD3'], dtype="int32")
    dev_ACS_labels= numpy.asarray(dev_data['ACS3'], dtype="int32")
    dev_HF_labels= numpy.asarray(dev_data['HF3'], dtype="int32")
    dev_AF_labels= numpy.asarray(dev_data['AF3'], dtype="int32")

    dev_CAD_OH_labels=to_categorical(((dev_CAD_labels+1)/1).astype(int))
    dev_ACS_OH_labels=to_categorical(((dev_ACS_labels+1)/1).astype(int))
    dev_HF_OH_labels=to_categorical(((dev_HF_labels+1)/1).astype(int))
    dev_AF_OH_labels=to_categorical(((dev_AF_labels+1)/1).astype(int))

    dev_CAD_BIN_labels=np.clip(np.absolute(dev_CAD_labels),0,1).astype(int)
    dev_ACS_BIN_labels=np.clip(np.absolute(dev_ACS_labels),0,1).astype(int)
    dev_HF_BIN_labels=np.clip(np.absolute(dev_HF_labels),0,1).astype(int)
    dev_AF_BIN_labels=np.clip(np.absolute(dev_AF_labels),0,1).astype(int)

    train_labels=[train_HF_OH_labels,train_HF_labels]#[train_CAD_OH_labels,train_ACS_OH_labels,train_HF_OH_labels,train_AF_OH_labels]
    dev_labels=[dev_HF_OH_labels,dev_HF_labels]#[dev_CAD_OH_labels,dev_ACS_OH_labels,dev_HF_OH_labels,dev_AF_OH_labels]

   # train_labels

    print("Parsing texts...")
    train_docs = list(nlp.pipe(train_data['note_text']))
    dev_docs = list(nlp.pipe(dev_data['note_text']))

    # TODO
    # if by_sentence:
    #     train_docs, train_labels = get_labelled_sentences(train_docs, train_labels)
    #     dev_docs, dev_labels = get_labelled_sentences(dev_docs, dev_labels)

    train_X = get_features(train_docs, max_length)
    dev_X = get_features(dev_docs, max_length)

    model = compile_lstm(
        embeddings,
        {"nr_hidden": nr_hidden, "max_length": max_length, "nr_class": 4},
        {"dropout": dropout, "lr": learn_rate}
    )
    cp_cb = ks.callbacks.ModelCheckpoint(filepath=os.path.join("./m",'LSTM_CNN_BEST_model.hdf5'), monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
    #rocab = classf.classf_callback(batch_size, training_data=train_X, training_ans=train_labels, validation_data=dev_X,validation_ans=dev_labels)
    #rocab=roc_auc.roc_callback(batch_size,training_data=train_X, training_ans=train_labels,validation_data=dev_X, validation_ans= dev_labels)
    rocab=corclass.corr_r_callback(batch_size,training_data=train_X, training_ans=train_labels,validation_data=dev_X, validation_ans= dev_labels)#'val_CAD_CAT_acc'
    callbacks=[rocab,cp_cb]
    model.load_weights(os.path.join("./PWCAD",'LSTM_CNN_BEST_model.hdf5'))
    #model.fit(train_X,train_labels,validation_data=(dev_X, dev_labels),epochs=nb_epoch,batch_size=batch_size,callbacks=callbacks)
    model.load_weights(os.path.join("./n",'LSTM_CNN_BEST_model.hdf5'))
    if return_keras_model:
        return model
    else:
        return LSTM_textcat(model,("CAD","ACS","HF","AF"))

def multi_conv(x,num_kernel,activation="relu"):
    kreg = None#regularizers.l2(0.01)
    a=Conv1D(num_kernel,3,activation=activation,padding="valid",kernel_regularizer=kreg)(x)
    #a=BatchNormalization()(a)
    b=Conv1D(num_kernel,3,activation=activation,padding="same",kernel_regularizer=kreg)(x)
    #b = BatchNormalization()(b)
    b = Conv1D(num_kernel, 3, activation=activation, padding="valid",kernel_regularizer=kreg)(b)
    #b = BatchNormalization()(b)


    #c=Conv1D(num_kernel,7,activation=activation,padding="same")(x)
    #c=BatchNormalization()(c)
    #d=Conv1D(num_kernel,9,activation=activation,padding="same")(x)
    #d=x=BatchNormalization()(d)
    return concatenate([a,b],axis=-1)
def compile_lstm(embeddings, shape, settings):
    print(shape)
    input1 = Input((shape["max_length"],))
    initial_kernel_num=64
    x = input1
    x = Embedding(embeddings.shape[0], embeddings.shape[1], input_length=shape["max_length"], trainable=False,
                  weights=[embeddings], mask_zero=False, )(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*2, return_sequences=True))(x)
    x = Conv1D(initial_kernel_num, 5, activation="relu", padding="valid")(x)
    x = multi_conv(x, initial_kernel_num)
    #x = Dropout(0.2)(x)
    # x=Bidirectional(LSTM(shape["nr_hidden"],recurrent_dropout=settings["dropout"],dropout=settings["dropout"],return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*2, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)

    #x = multi_conv(x, initial_kernel_num*2)
    x = multi_conv(x, initial_kernel_num*2)
    #x = Dropout(0.2)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*4, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)



    x = multi_conv(x, initial_kernel_num*4)
    # x=Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)


    x = multi_conv(x, initial_kernel_num*4)
    # x=Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = multi_conv(x, initial_kernel_num*4)
    # x=Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = multi_conv(x, initial_kernel_num * 4)
    # x=Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = multi_conv(x, initial_kernel_num * 4)
    # x=Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num * 8, return_sequences=True))(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = multi_conv(x, initial_kernel_num*4)
    # x=Dropout(0.1)(x)
    x = Bidirectional(CuDNNLSTM(initial_kernel_num*8, return_sequences=False))(x)
    #x = Flatten()(x)
    #x = Dropout(0.5)(x)
    #x = Dense(512, activation="relu")(x)
    #x = BatchNormalization()(x)
    #x = Dropout(0.5)(x)


    a1=Dense(512,activation="relu")(x)
    #a1=BatchNormalization()(a1)
    a1=Dropout(0.5)(a1)
    a1=Dense(3, activation="softmax",name="CAD_CAT")(a1)


    a2 = Dense(512, activation="relu")(x)
    #a2 = BatchNormalization()(a2)
    a2 = Dropout(0.5)(a2)
    a2 = Dense(1, name="CAD_REG")(a2)
    # b=Dense(3, activation="softmax",name="ACS")(x)
    # c=Dense(3, activation="softmax",name="HF")(x)
    # d=Dense(3, activation="softmax",name="AF")(x)
    model = Model(inputs=input1, outputs=[a1,a2])
    weights = np.array([0.63, 0.33, 0.63])#,weighted_categorical_crossentropy(weights)
    loss=[weighted_categorical_crossentropy(weights),"mean_squared_error"] #"categorical_crossentropy"#'binary_crossentropy'#["categorical_crossentropy","mean_squared_error"]
    loss_weights=[0.4,0.6]
    model.compile(optimizer=RMSprop(lr=settings["lr"]),loss=loss,metrics=["accuracy"],loss_weights=loss_weights)#(optimizer=Adam(lr=settings["lr"]),loss="binary_crossentropy",metrics=["accuracy"],)


    model.summary()
    return model


def get_embeddings(vocab):
    return vocab.vectors.data


# TODO
def predict(model_path, text_paths, output_path=None):
    """Load model from disk, read files from disk and output predictions."""
    # TODO
    pass

    # nlp = spacy.load("en_vectors_web_lg")
    # nlp.add_pipe(nlp.create_pipe("sentencizer"))
    # nlp.add_pipe(SentimentAnalyser.load(model_dir, nlp, max_length=max_length))

    # import pandas as pd
    #
    # print("Loading model...")
    # nlp = spacy.load(model_path)
    # textcat = nlp.get_pipe('textcat')
    # print("Predicting...")
    # predictions = _predict(nlp.tokenizer,
    #                        textcat,
    #                        textcat.read_json_data(text_paths, shuffle=False))
    #
    # predictions = pd.DataFrame(predictions)
    # if output_path is None:
    #     # create a default output name
    #     output_path = text_paths[0] + '-predictions.csv'
    # output_path = srsly.util.force_path(output_path, require_exists=False)
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # print("Writing output...")
    # predictions.to_csv(output_path)


def _predict(pipe, data, nlp_str="en_vectors_web_lg"):
    """Predict using a tokenizer and model object on loaded data."""

    nlp = spacy.load(nlp_str)
    nlp.add_pipe(pipe)

    start_time = time.time()
    predictions = [doc.user_data['my_cats'] for doc in nlp.pipe((row['text'] for row in data), batch_size=500)]

    if 'noteid' in data[0]:
        predictions = [dict(note_id=k, **p) for k, p in zip((row['noteid'] for row in data), predictions)]
    end_time = time.time()
    print("--- prediction time: %s seconds ---" % (end_time - start_time))
    return predictions


def evaluate_textcat(pipe, eval_data, nlp_str="en_vectors_web_lg"):
    nlp = spacy.load("/mnt/DataA/TextClassifiyer/clinical_notes2/data/word_vectors/fasttext/result_spacy/all_notes_5_12_19-default_epochs20")
    #nlp = spacy.load("en_vectors_web_lg")#('/mnt/DataA/TextClassifiyer/clinical_notes/data/fasttext/en_vectors_web_lg')#("en_vectors_web_lg")\
    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    #nlp.add_pipe(nlp.create_pipe("sentencizer"))
    embeddings = get_embeddings(nlp.vocab)
    eval_labels=numpy.asarray(eval_data['HF3'], dtype="int32")
    #eval_OH_labels=to_categorical((eval_labels).astype(int)+2)
    eval_OH_labels = to_categorical((eval_labels).astype(int)+1 )
    eval_BIN_labels=np.clip(np.absolute(eval_labels),0,1).astype(int)
    eval_docs =  get_features(list(nlp.pipe(eval_data['note_text'])),1500)
    predictions = pipe.predict(eval_docs,batch_size=256)
    y_pred_val=predictions[0]
    f1_val=precision_recall_fscore_support(np.argmax(eval_OH_labels,axis=-1),np.argmax(y_pred_val,axis=-1), average="macro")
    #f1_val=precision_recall_fscore_support(eval_labels+1,(np.clip(y_pred_val+1.5,0,2)).astype(int), average="macro")
    print(f1_val)
    #print(confusion_matrix(eval_labels+1,(np.clip(y_pred_val+1.5,0,2)).astype(int)))
    print(confusion_matrix(np.argmax(eval_OH_labels,axis=-1),np.argmax(y_pred_val,axis=-1)))
    #roc=roc_auc_score(eval_BIN_labels,y_pred_val)
    #print(roc)
    eval_data["HF_Predictions"]=(np.argmax(y_pred_val,axis=-1)-1)
    eval_data["HF_Raw_NEG"]=y_pred_val[:,0]
    eval_data["HF_Raw_UI"]=y_pred_val[:,1]
    eval_data["HF_Raw_POS"]=y_pred_val[:,2]
    #eval_data["Predictions"]=(np.clip(y_pred_val+1.5,0,2)).astype(int)
    #eval_data["Pred_Raw_Score"]=y_pred_val
    #eval_data["Informative"]=eval_OH_labels
    eval_data.to_excel("output.xlsx")

if __name__ == "__main__":
    # TODO
    #plac.call(main)
    pass


# %%

train_paths = ['/mnt/DataA/TextClassifiyer/clinical_notes2/data/labels/7_7_19/annotated_feather/1.feather',
               '/mnt/DataA/TextClassifiyer/clinical_notes2/data/labels/7_7_19/annotated_feather/2.feather',
               '/mnt/DataA/TextClassifiyer/clinical_notes2/data/labels/7_7_19/annotated_feather/3.feather']
dev_paths = ['/mnt/DataA/TextClassifiyer/clinical_notes2/data/labels/7_7_19/annotated_feather/4.feather']

# %%
lstm_textcat_model = train(train_paths, dev_paths, nb_epoch=150)#150)
#predictions = _predict(lstm_textcat_pipe, textcat.read_json_data(dev_paths))
scores = evaluate_textcat(lstm_textcat_model,pd.read_feather('/mnt/DataA/TextClassifiyer/clinical_notes2/data/labels/7_7_19/annotated_feather/5.feather'))
print(scores)
#lstm_textcat_pipe.save("/home/mhomilius/clinical_notes/results/models/keras_lstm_test")
# TODO try loading and predicting from loaded model
#lstm_textcat_pipe.save("/home/mhomilius/clinical_notes/results/models/keras_lstm_test")
