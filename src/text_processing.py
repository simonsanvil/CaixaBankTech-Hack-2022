import pandas as pd

def tokenize(x: pd.Series, nlp):
    """
    Tokeniza una serie de texto.
    """
    return x.apply(lambda s: nlp(s))

def drop_symbols(x:pd.Series):
    """
    Elimina los simbolos de una serie de texto removiendo todos los caracteres que no sean alfanumericos.
    """
    return x.apply(lambda doc: [token for token in doc if token.is_alpha])

def drop_stopwords(x:pd.Series):
    """
    Elimina las stopwords de una serie de texto removiendo todos los caracteres que no sean alfanumericos.
    """
    return x.apply(lambda doc: [token for token in doc if not token.is_stop])

def lemmatize(x:pd.Series):
    """
    Lematiza una serie de texto para dejar solo las palabras relevantes.
    """
    return x.apply(lambda doc: ' '.join(token.lemma_ for token in doc))

def filter_pos(x:pd.Series, pos:set):
    """
    Filtra las palabras de una serie de texto manteniendo solo las palabras que pertenecen a un determinado conjunto de Part-Of-Speech.
    """
    return x.apply(lambda doc: [token for token in doc if token.pos_ in pos])

def detokenize(x:pd.Series):
    """
    Reemplaza los espacios en blanco por un espacio.
    """
    return x.apply(lambda doc: ' '.join(token.text for token in doc))
