import os
import gc
import datetime
import numpy as np
import pandas as pd
import itertools
import collections
import logging
import glob

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, concatenate, GRU, Embedding, Flatten
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras import backend as K
from keras import optimizers
from keras import losses

'''

TODO :
    emoji(binary) preprocessing / Stemmer
    input text direction / bi-directional
    mask zero
    cutting
    different scale target
    dense regularizer
    sub category
    remove price ==0 / price < 1 from training set, but still stimulate with
    share representation net ensemble
    not_desc fill 
    
    brand : ALL, X, M => missing?

'''


def get_input_data(datafolder):
    filenames = os.listdir(datafolder)
    data = dict()
    for i in filenames:
        topic, *mid, filetype = i.split('.')
        if filetype == 'tsv':
            data[topic] = pd.read_table(datafolder + i)
        elif filetype == 'csv':
            data[topic] = pd.read_csv(datafolder + i)
        else:
            pass

    sub = data['sample_submission']
    train = data['train']
    test = data['test']

    return train, test, sub


def train_test_combine(train, test):
    train.rename(columns={'train_id': 'id'}, inplace=True)
    test.rename(columns={'test_id': 'id'}, inplace=True)
    train['is_train'] = 1
    test['is_train'] = 0
    df = pd.concat([train, test], ignore_index=True)
    return df


def fillna_with_other_column(df, to_fill='item_description', material='name'):
    to_fill_isna = (df[to_fill].isna() | (df[to_fill] == 'No description yet'))
    df.loc[to_fill_isna, to_fill] = df[material][to_fill_isna]
    return df


def fillna_by_mode_of_shared_value_of_other_column(df, to_fill='category_name', material='name'):
    category_na = df[to_fill].isna()
    groups = [v for k, v in df.groupby(material).groups.items() if (
        (len(v) > 1) and
        (df[to_fill].iloc[v].isna().sum()) and
        (df[to_fill].iloc[v].notna().sum()))]

    for i in groups:
        df.loc[category_na & df.index.isin(i), to_fill] = df.iloc[i][to_fill].dropna().mode().values[0]
    return df


def fillna_category_from_other(df, to_fill='brand_name', material='name'):
    brands = df[to_fill].dropna().drop_duplicates()
    brand_counter = CountVectorizer()
    brand_counter.fit(brands)
    brand_token = (brand_counter.transform(brands).T / brands.apply(lambda x: len(x.split())).values)
    brandna = df[to_fill].isna()
    name_trans = brand_counter.transform(df[material][brandna].values)
    na_brands = list()
    for i, r in enumerate(name_trans):
        try:
            na_brands.append(brands.iloc[np.where(np.dot(r.toarray().reshape(1, -1), brand_token) == 1)[1][0]])
        except IndexError:
            na_brands.append(np.NAN)
    df.loc[brandna, to_fill] = na_brands
    return df


def train_test_recover(df):
    train = df[df['is_train'] == 1].reset_index()
    test = df[df['is_train'] == 0].reset_index()
    del train['is_train']
    del test['is_train']
    return train, test


def simulate_test(df, simulate_count=2800000):
    indices = np.random.choice(df.index.values, simulate_count)
    df_ = pd.concat([df, df.iloc[indices]], axis=0)
    return df_.copy()


def na_fill_missing(df, fill_value='missing'):
    df.category_name.fillna(value=fill_value, inplace=True)
    df.brand_name.fillna(value=fill_value, inplace=True)
    df.item_description.fillna(value=fill_value, inplace=True)
    return df


def categorical_transform(train, test, to_column='category', from_column='category_name'):
    le = LabelEncoder()
    le.fit(np.hstack([train[from_column], test[from_column]]))
    train[to_column], test[to_column] = map(le.transform, [train[from_column], test[from_column]])
    return train, test


def text2seq(train, test, tokenizer, cols=('category_name', 'item_description', 'name')):

    for df, col in itertools.product([train, test], cols):
        seq_col = 'seq_{}'.format(col)
        df[seq_col] = tokenizer.texts_to_sequences(df[col].str.lower())
    return train, test


def get_keras_data(df):
    keras_data = {
        'name': pad_sequences(df['seq_name'], maxlen=MAX_NAME_SEQ),
        'item_desc': pad_sequences(df['seq_item_description'], maxlen=MAX_ITEM_DESC_SEQ),
        'brand': np.array(df['brand']),
        'category': np.array(df['category']),
        'category_name': pad_sequences(df['seq_category_name'], maxlen=MAX_CATEGORY_NAME_SEQ),
        'item_condition': np.array(df['item_condition_id']),
        'shipping': np.array(df[["shipping"]])
    }
    return keras_data


def get_text_tokenizer(train, test, cols=('category_name', 'item_description', 'name')):
    text_tokenizer = Tokenizer()
    text_tokenizer.fit_on_texts(
        np.hstack([df[col].astype('str').str.lower() for df, col in itertools.product([train, test], cols)])
    )
    return text_tokenizer


def get_glove(glove_path = 'D:/Users/shanger_lin/Desktop/jupyter_working/NLP projects/embedding/glove.6B/glove.6B.50d.txt',
              check_length=100):

    c = collections.Counter()
    with open(glove_path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i >= check_length:
                break
            c.update((len(line.split()),))
    line_len = c.most_common(1)[0][0]

    with open(glove_path, 'r', encoding='utf8') as f:
        vocab = list()
        embedding = list()
        glove_word_vectors = dict()
        for line in f:

            splitline = line.split()

            if len(splitline) != line_len:
                logging.warning('skip a word {}, which is with unmatched length.'.format(splitline[0]))
                continue

            word = splitline[0]
            vector = list(map(float, splitline[1:]))

            vocab.append(word)
            embedding.append(vector)
            glove_word_vectors[word] = vector

    embedding = np.array(embedding, dtype=np.float32)
    return vocab, embedding, glove_word_vectors


def get_model(embedding_matrix):
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand = Input(shape=[1], name="brand")
    category = Input(shape=[1], name="category")
    category_name = Input(shape=[X_train["category_name"].shape[1]], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    shipping = Input(shape=[X_train["shipping"].shape[1]], name="shipping")

    vocab_size, emb_size = embedding_matrix.shape
    glove_name = Embedding(vocab_size, emb_size, trainable=GLOVE_TRAINABLE, weights=[embedding_matrix], mask_zero=True)(name)
    glove_item_desc = Embedding(vocab_size, emb_size, trainable=GLOVE_TRAINABLE, weights=[embedding_matrix], mask_zero=True)(item_desc)
    glove_category_name = Embedding(vocab_size, emb_size, trainable=GLOVE_TRAINABLE, weights=[embedding_matrix], mask_zero=True)(category_name)

    emb_name = Embedding(MAX_TEXT, 20, mask_zero=True)(name)
    emb_item_desc = Embedding(MAX_TEXT, 60, mask_zero=True)(item_desc)
    emb_category_name = Embedding(MAX_TEXT, 20, mask_zero=True)(category_name)

    emb_brand = Embedding(MAX_BRAND, 10)(brand)
    emb_category = Embedding(MAX_CATEGORY, 10)(category)
    emb_item_condition = Embedding(MAX_CONDITION, 5)(item_condition)

    emb_item_desc = GRU(16)(emb_item_desc)
    emb_category_name = GRU(8)(emb_category_name)
    emb_name = GRU(8)(emb_name)

    glove_item_desc = GRU(16)(glove_item_desc)
    glove_category_name = GRU(8)(glove_category_name)
    glove_name = GRU(8)(glove_name)

    main_l = concatenate([
        Flatten()(emb_brand),
        Flatten()(emb_category),
        Flatten()(emb_item_condition),
        # glove_item_desc,
        # glove_category_name,
        # glove_name,
        emb_item_desc,
        emb_category_name,
        emb_name,
        shipping,
    ])
    # main_l = Dropout(0.5)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(0.3)(Dense(512, activation='relu')(main_l))
    main_l = Dropout(0.2)(Dense(64, activation='relu')(main_l))
    output = Dense(1, activation="linear")(main_l)

    model = Model([name, item_desc, brand, category, category_name, item_condition, shipping], output)
    optimizer = optimizers.Adam()
    loss = losses.mean_squared_error
    model.compile(loss=loss, optimizer=optimizer)
    return model


def get_embedding_matrix(word_index, word_to_vector):
    embed_dimension = len(tuple(word_to_vector.values())[0])
    embedding_matrix = np.zeros((len(word_index) + 1, embed_dimension))
    for word, i in word_index.items():
        embedding_vector = word_to_vector.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def timer():
    now = datetime.datetime.now()
    start = now
    last = now
    step = 0
    print('[START]', now.isoformat())
    message = yield None

    while True:
        if message:
            message = str(message)
        else:
            message = ''

        step += 1
        now = datetime.datetime.now()
        from_start = (now - start).total_seconds()
        from_last = (now - last).total_seconds()
        print('[{}.{}] : {}, which is {}sec from start, {}sec from last'.format(
            step, message, now.isoformat(), from_start, from_last))
        last = now
        message = yield None


def fill_na_randomly(to_fill_df, filled_df, fill_cols=('item_description', 'category_name', 'brand_name')):
    fill_message = list()

    for fill_col in fill_cols:
        if np.random.ranf(1)[0] > 0.5:
            to_fill_df[fill_col] = filled_df[fill_col]
            fill_message.append(fill_col)
    if fill_message:
        fill_message = '_'.join(fill_message)
    else:
        fill_message = 'None'
    return to_fill_df, fill_message

if __name__ == '__main__':

    timer = timer()
    next(timer)

    if np.random.ranf(1)[0] > 0.2:
        GLOVE_TRAINABLE = True
    else:
        GLOVE_TRAINABLE = False
    print('GLOVE_TRAINABLE', GLOVE_TRAINABLE)

    DATAFOLDER = '../input/mercari-price-suggestion-challenge/'
    train_df, test_df, sub_df = get_input_data(DATAFOLDER)


    timer.send('data read')
    combine_df = train_test_combine(train_df, test_df)

    # combine_df = fillna_with_other_column(combine_df, to_fill='item_description', material='name')
    # combine_df = fillna_by_mode_of_shared_value_of_other_column(combine_df, to_fill='category_name', material='name')
    # combine_df = fillna_category_from_other(combine_df, to_fill='brand_name', material='name')

    # filled_file = 'combined_na_filled_v3.csv'
    # filled_df = pd.read_csv(filled_file)
    # combine_df, fill_message = fill_na_randomly(
    #     combine_df, filled_df, fill_cols=('item_description', 'category_name', 'brand_name'))
    fill_message = 'None'
    train_df, test_df = train_test_recover(combine_df)
    print('fill_message', fill_message)
    timer.send('na filled')

    original_test_row = test_df.shape[0]

    train_df['target'] = np.log1p(train_df['price'])

    simulate_message = 'notsimulate'
    if np.random.ranf(1)[0] > 0.5:
        test_df = simulate_test(test_df)
        simulate_message = 'simulate'
    print('simulate_message', simulate_message)
    timer.send('simulate_test')

    train_df, test_df = map(na_fill_missing, [train_df, test_df])
    timer.send('na_fill_missing')
    text_tokenizer = get_text_tokenizer(train_df, test_df, cols=('category_name', 'item_description', 'name'))
    timer.send('get_text_tokenizer')
    # TODO : is it necessary to fit tokenizer with test set?  #  A: yes for pre-trained model but slow
    train_df, test_df = categorical_transform(train_df, test_df, to_column='category', from_column='category_name')
    train_df, test_df = categorical_transform(train_df, test_df, to_column='brand', from_column='brand_name')
    timer.send('categorical_transform')
    train_df, test_df = text2seq(train_df, test_df, text_tokenizer, cols=('category_name', 'item_description', 'name'))
    timer.send('text2seq')

    MAX_NAME_SEQ = 20  # 7
    MAX_ITEM_DESC_SEQ = 60  # 269
    MAX_CATEGORY_NAME_SEQ = 20  # 8
    MAX_BRAND = combine_df['brand_name'].drop_duplicates().count()+1
    MAX_CONDITION = combine_df['item_condition_id'].drop_duplicates().count()+1
    MAX_CATEGORY = combine_df['category_name'].drop_duplicates().count()+1
    MAX_TEXT = len(text_tokenizer.word_index)

    del combine_df, train_df['brand_name'], test_df['brand_name']
    timer.send('parameter_set')

    dtrain, dvalid = train_test_split(train_df, train_size=0.9)
    timer.send('train_test_split')
    X_train, X_valid, X_test = map(get_keras_data, [dtrain, dvalid, test_df[:original_test_row]])
    timer.send('get_keras_data')

    exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
    epochs = 2
    BATCH_SIZE = 512 * 3
    steps = len(X_train)//BATCH_SIZE * 3  #  epochs  # train_df.shape[0] will not be the same as X_train
    lr_init, lr_fin = 0.009, 0.006
    lr_decay = exp_decay(lr_init, lr_fin, steps)
    log_subdir = '_'.join(['ep', str(epochs),
                           'bs', str(BATCH_SIZE),
                           'lrI', str(lr_init),
                           'lrF', str(lr_fin),
                           'dr', str(lr_decay)])

    import glob
    glove_path = np.random.choice(glob.glob('../input/glove-wiki-twitter2550/*.txt')).replace('\\', '/')
    glove = glove_path.split('/')[-1].replace('.txt', '')
    print('glove', glove)
    _, _, word_to_vector = get_glove(glove_path=glove_path)
    embedding_matrix = get_embedding_matrix(text_tokenizer.word_index, word_to_vector)

    model = get_model(embedding_matrix)

    K.set_value(model.optimizer.lr, lr_init)
    K.set_value(model.optimizer.decay, lr_decay)

    del _
    gc.collect()

    timer.send('training_prepared')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')
    history = model.fit(X_train, dtrain.target,
                        epochs=epochs,
                        batch_size=BATCH_SIZE,
                        validation_split=0.05,
                        verbose=1,
                        shuffle=True,
                        callbacks=[earlyStopping],
                        )
    timer.send('trained')

    train_preds = model.predict(X_valid, batch_size=BATCH_SIZE)
    validation_rmsle = np.sqrt(mean_squared_error(train_preds, dvalid.target))
    timer.send('valid')
    print('validation_rmsle', validation_rmsle)

    preds = np.expm1(model.predict(X_test, batch_size=BATCH_SIZE))
    timer.send('predicted')

    test_df.rename(columns={'id': 'test_id'}, inplace=True)
    submission = test_df[["test_id"]][:original_test_row]
    submission["price"] = preds
    now = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    message = 'justGlove'

    submit_name = "{}_GRU_{}_{}_{}_{}_{}.csv".format(
        now, simulate_message, fill_message, '{}{}'.format(glove, GLOVE_TRAINABLE), validation_rmsle, message)
    submission.to_csv('{}'.format(submit_name), index=False)
    timer.send(submit_name)

pass
