# NER_Drug_Prescriptions

A solution for the TAC 2017 - Adverse Drug Reaction Extraction from Drug Labels.

## Architecture

Model implementation present in [main_tf.py](./models/chars_conv_lstm_crf/main_tf.py)
1. Glove Embeddings for words
2. Character Embeddings
3. 1d convolution and max pooling on Character Embeddings
4. Bi-LSTM
5. CRF

### Data

The data files are downloaded from [here](https://bionlp.nlm.nih.gov/tac2017adversereactions/)

Download glove embeddings from [here](https://nlp.stanford.edu/projects/glove/).
Extract it inside the data folder.

### Scripts

1. [annotate_data.py](./scripts/annotate_data.py) - Preprocess and tag data.
2. [build_vocab.py](./scripts/build_vocab.py) - Build the vocabulary
3. [build_glove.py](./scripts/build_glove.py) - Retrieve glove embeddings for each word.
