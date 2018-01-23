import re

tokenization_regex = r'[^\s][-a-zA-Z0-9]*[^\s]?'


def scrm114_tokenizer(in_string):
    return re.findall(tokenization_regex, in_string)


def eager_split_tokenizer(in_string):
        return re.split(r'[\s\.,:-]', in_string)
