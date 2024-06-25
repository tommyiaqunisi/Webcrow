import os.path

import numpy as np
import time
import pandas as pd
import dill as pickle
from experts.neuralsearch.common import text_cleaning
from gensim.models import KeyedVectors
from experts.neuralsearch.models import ModelsFactory

# SET HERE CONFIG OPTIONS
#WORD2VEC_GOOGLE_NEWS_PATH = "./data/embedding_models/itwiki_20180420_300d.txt"
MIN_ANSWER_LEN = 3     # Answers with less than 3 characters are noise. 2 FOR ITALIAN
LANG = 'EN' #IT or EN
MODEL_NAME = "EN-USE"  # 
PATH = "./data/CWDB/"

def add_case_insensitive_embeddings_to_word2vec(word2vec_model):
    wv_to_add = dict()
    in_vocab_words = frozenset(word2vec_model.vocab.keys())
    for w in in_vocab_words:
        lowercased = w.casefold()
        if lowercased not in in_vocab_words:
            wv_to_add[lowercased] = word2vec_model.get_vector(w)

    new_words, new_vectors = zip(*wv_to_add.items())
    # Gensim expects two lists here.
    word2vec_model.add(list(new_words), list(new_vectors))


def has_any_in_vocab_tokens(document_tokens, vocab_tokens):
    return any([True if tok in vocab_tokens else False for tok in document_tokens])


def clean_up_df(df, mode, word2vec_vectors_num=250000):
    # Very few rows contain "nil" or "null" in their answers.
    df = df.dropna().copy()

    # Save ourselves from encoding headaches by keeping ASCII rows only.
    # Moreover, we don't really need rows with questions having digits, so we drop them.
    #df.clue = df.clue.apply(lambda x: x if text_cleaning.is_ascii(x) else None)
    # df.clue = df.clue.apply(lambda x: x if text_cleaning.has_no_digits(x) else None)
    df.answer = df.answer.apply(lambda x: x if text_cleaning.is_ascii(x) else None)

    df.clue = df.clue.apply(text_cleaning.strip_punctuation)
    df.answer = df.answer.apply(text_cleaning.strip_punctuation)

    df = df[df.answer.str.len() >= MIN_ANSWER_LEN].copy()

    # We don't need rows with NaNs
    df = df.replace("", pd.NA)
    df = df.dropna().copy()

    df.clue = df.clue.apply(text_cleaning.to_lowercase)
    df.answer = df.answer.apply(text_cleaning.to_lowercase)

    # if word2vec_vectors_num > 0:
    #     # Apply same splitting strategy as in `neuralsearch.domain.document._words_tokenize` method.
    #     df["question_tokens"] = df.clue.str.split()
    #     df["answer_tokens"] = df.answer.str.split()

    #     word2vec = KeyedVectors.load_word2vec_format(
    #         WORD2VEC_GOOGLE_NEWS_PATH,
    #         binary=False,
    #         limit=word2vec_vectors_num,
    #     )
    #     add_case_insensitive_embeddings_to_word2vec(word2vec)
    #     model_in_vocab_words = frozenset(word2vec.vocab.keys())
    #     # Drop the rows for which all document's words are Out-of-Word2vec-Vocabulary.
    #     if mode == "QQ":
    #         df = df[df["question_tokens"].apply(has_any_in_vocab_tokens, args=(model_in_vocab_words,))]
    #     else:
    #         df = df[df["answer_tokens"].apply(has_any_in_vocab_tokens, args=(model_in_vocab_words,))]

    # print(df.info())
    # df = df[df["answer"].map(df["answer"].value_counts()) != 1]
    print(df.info())

    return df





if __name__ == "__main__":
    # embed models
    #embedding_model = models_factory(model_name=MODEL_NAME, lang=LANG)
    embedding_model = ModelsFactory.from_pretrained(MODEL_NAME)
    embedding_model.load_model()
    data_base_path = PATH

    # data_qa_path = os.path.join(data_base_path, "QA")
    # if not os.path.exists(data_qa_path):
    #     os.makedirs(data_qa_path)

    modes = ["QQ"]
    # answer_lens = range(3, 25)  # todo change here for different size
    answer_lens = range(3, 25)  # todo change here for different size
    for mode in modes:
        data_mode_path = os.path.join(data_base_path, mode)
        if not os.path.exists(data_mode_path):
            os.makedirs(data_mode_path)

        for answer_len in answer_lens:
            print("====================================================")
            print(f"================ PROCESSING {mode} - {answer_len} ================")
            # load clue-answer csv
            data_path = os.path.join(data_base_path, f"train/train_length_{answer_len}.csv")
            if os.path.exists(data_path):
                train_df = pd.read_csv(data_path, sep=",")
                train_df = clean_up_df(train_df, mode=mode)
                if train_df.shape[0] > 0:
                    print("Encoding Questions and Answers")
                    start_time = time.time()
                    questions, answers, q_embeddings, a_embeddings = [], [], [], {}
                    n_occurrencies = []
                    for q, a, n_occ in list(zip(train_df.clue, train_df.answer, train_df.couple_occurencies)):
                        # computing all the embeddings and the rest of needed features
                        questions.append(q)
                        answers.append(a)
                        n_occurrencies.append(n_occ)
                        if mode == "QQ":
                            q_embeddings.append(embedding_model.embed([q]).astype(np.float16).squeeze())
                            #q_embeddings.append(embedding_model.embed(q).astype(np.float16))
                        if a not in a_embeddings:
                            a_embeddings[a] = embedding_model.embed([a]).astype(np.float16).squeeze()
                            #a_embeddings[a] = embedding_model.embed(q).astype(np.float16)
                    print(f"--- Encoding took: {((time.time() - start_time) * 1000)} ms ---")
                    # saving
                    embeddings = q_embeddings if mode == "QQ" else a_embeddings
                    df = (questions, answers, embeddings, n_occurrencies)
                    f_path = os.path.join(data_mode_path, f"{answer_len}_{LANG}_{MODEL_NAME}_{mode}.pkl")
                    f = open(f_path, "wb")
                    pickle.dump(df, f)
                    f.close()
                else:
                    print(f"Csv at {data_path} was empty or NO Clue-Answer pair had values in Word2Vec Vocabulary")
            else:
                print(f"{data_path} Not found. Skipping it")



