import pandas as pd
import numpy as np
from urllib.parse import urlparse
import tldextract
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

#Lexical features 
SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'account', 'secure', 'signin', 'banking', 
    'paypal', 'ebay', 'chase', 'admin', 'support', 'update', 'webscr'
]

SHORTENING_SERVICES = [
    'bit.ly', 't.co', 'goo.gl', 'tinyurl', 'ow.ly', 'is.gd', 'buff.ly'
]

def extract_lexical_features(url):
    """returns dict of features"""
    features = {}
    if not re.match(r'^(http|https|ftp)://', url):
        url = 'http://' + url

    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname if parsed_url.hostname else ''
        path = parsed_url.path if parsed_url.path else ''
        query = parsed_url.query if parsed_url.query else ''
        tld_parts = tldextract.extract(url)
        domain = tld_parts.domain
        subdomain = tld_parts.subdomain
        tld = tld_parts.suffix     
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
        #all NaN dict if error
        return {f: np.nan for f in [
            'url_length', 'hostname_length', 'path_length', 'tld_length',
            'count_hyphen', 'count_at', 'count_question', 'count_percent',
            'count_dot', 'count_equals', 'count_double_slash', 'count_digits',
            'count_letters', 'count_subdomains', 'is_ip_address', 'is_https',
            'is_shortened', 'has_suspicious_keyword', 'abnormal_subdomain',
            'tld_in_path', 'tld_in_subdomain'
        ]}

    #1.Len features 
    features['url_length'] = len(url)
    features['hostname_length'] = len(hostname)
    features['path_length'] = len(path)
    features['tld_length'] = len(tld)

    #2.Charcnt features
    features['count_hyphen'] = url.count('-')
    features['count_at'] = url.count('@')
    features['count_question'] = url.count('?')
    features['count_percent'] = url.count('%')
    features['count_dot'] = url.count('.')
    features['count_equals'] = url.count('=')
    features['count_double_slash'] = url.count('//')
    features['count_digits'] = sum(c.isdigit() for c in url)
    features['count_letters'] = sum(c.isalpha() for c in url)
    features['count_subdomains'] = len(subdomain.split('.')) if subdomain else 0

    #3.bool features
    ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    features['is_ip_address'] = 1 if re.match(ip_pattern, hostname) else 0
    features['is_https'] = 1 if parsed_url.scheme == 'https' else 0
    features['is_shortened'] = 1 if any(service in url for service in SHORTENING_SERVICES) else 0
    features['has_suspicious_keyword'] = 1 if any(keyword in url.lower() for keyword in SUSPICIOUS_KEYWORDS) else 0

    
    #4.other features 
    features['abnormal_subdomain'] = 1 if (domain and subdomain.count(domain) > 0) else 0
    features['tld_in_path'] = 1 if (tld and tld in path) else 0
    features['tld_in_subdomain'] = 1 if (tld and tld in subdomain) else 0

    return features


#url character sequence
def create_url_tokenizer():
    tokenizer = Tokenizer(char_level=True, oov_token='UNK')
    Alphabet = ("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "0123456789.;,!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}")
    
    char_dict = {}
    for i, char in enumerate(Alphabet):
        char_dict[char] = i + 1
    tokenizer.word_index = char_dict
    max_value = max(char_dict.values())
    tokenizer.word_index[tokenizer.oov_token] = max_value + 1
    
    return tokenizer

def process_urls_to_sequences(url_list, tokenizer, max_len=200):
    sequences = tokenizer.texts_to_sequences(url_list)
    F_U = pad_sequences(sequences, 
                          maxlen=max_len, 
                          padding='post',   
                          truncating='post') 
    return F_U

if __name__ == "__main__":
    with open("data.pkl", "rb") as f:
        data = pickle.load(f)
    df_input = pd.DataFrame(data)
    df_lexical = df_input['url'].apply(lambda x: pd.Series(extract_lexical_features(x)))

    url_tokenizer = create_url_tokenizer()
    char_seq_vectors = process_urls_to_sequences(df_input['url'], url_tokenizer, max_len=200)
    char_seq_columns = [f'c_{i+1}' for i in range(200)]
    df_char_seq = pd.DataFrame(char_seq_vectors, columns=char_seq_columns, dtype=int)

    df_final = pd.concat([
        df_input.reset_index(drop=True),
        df_lexical.reset_index(drop=True),
        df_char_seq.reset_index(drop=True)
    ], axis=1)
    output_csv_filename = './input_data/url_features_extracted.csv'
    df_final.to_csv(output_csv_filename, index=False)
    print(f"All {df_final.shape[1]} features saved to '{output_csv_filename}'")