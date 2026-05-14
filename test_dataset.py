import sys
import logging
from transformers import DistilBertTokenizerFast
from src.config import TRAIN_CSV, MAX_SEQ_LEN, PRETRAINED_MODEL
from src.dataset import WELFakeDataset

print('Loading tokenizer...')
tokenizer = DistilBertTokenizerFast.from_pretrained(PRETRAINED_MODEL)

print('Tokenizer loaded. Creating train_ds...')
try:
    train_ds = WELFakeDataset(TRAIN_CSV, tokenizer, MAX_SEQ_LEN)
    print(f'Train dataset created! Length: {len(train_ds)}')
except Exception as e:
    print(f'Error creating train_ds: {e}')

print('Script finished successfully.')
