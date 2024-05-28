import clip
import torch
import array
import os
from tqdm import tqdm
import zipfile
import six
import torch
from six.moves.urllib.request import urlretrieve

def obj_edge_vectors(names, wv_dir='', wv_type='glove', wv_dim=300, use_cache=False):
    if 'glove' in wv_type:
        wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

        vectors = torch.Tensor(len(names), wv_dim)
        vectors.normal_(0,1)

        for i, token in enumerate(names):
            wv_index = wv_dict.get(token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                # Try the longest word
                lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
                # print("{} -> {} ".format(token, lw_token))
                wv_index = wv_dict.get(lw_token, None)
                if wv_index is not None:
                    vectors[i] = wv_arr[wv_index]
        return vectors
    elif wv_type == 'clip':
        # check cache
        if use_cache:
            cache_file = os.path.join(wv_dir, wv_type + '_obj.pt')
            if os.path.exists(cache_file):
                txt_feats = torch.load(cache_file)
                if len(names) == txt_feats.size(0):
                    return txt_feats

        model = clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(names).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(80)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)

        del model
        torch.cuda.empty_cache()

        # saving to cache
        if use_cache:
            if not os.path.exists(wv_dir):
                os.makedirs(wv_dir)
            torch.save(txt_feats, cache_file)

    return txt_feats

def rel_vectors(names, wv_dir='', wv_type='clip', wv_dim=300, use_cache=False):
    if 'glove' in wv_type:
        wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

        vectors = torch.Tensor(len(names), wv_dim)  # 51, 200
        vectors.normal_(0, 1)
        for i, token in enumerate(names):
            if i == 0:
                continue
            wv_index = wv_dict.get(token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                split_token = token.split(' ')
                ss = 0
                s_vec = torch.zeros(wv_dim)
                for s_token in split_token:
                    wv_index = wv_dict.get(s_token)
                    if wv_index is not None:
                        ss += 1
                        s_vec += wv_arr[wv_index]
                    else:
                        print("fail on {}".format(token))
                s_vec /= ss
                vectors[i] = s_vec

        return vectors
    elif wv_type == "clip":
        # check cache
        if use_cache:
            cache_file = os.path.join(wv_dir, wv_type + '_rel.pt')
            if os.path.exists(cache_file):
                txt_feats = torch.load(cache_file)
                if len(names) == txt_feats.size(0):
                    return txt_feats
        
        model = clip.load("ViT-B/32")[0]
        device = next(model.parameters()).device
        text_token = clip.tokenize(names).to(device)
        txt_feats = [model.encode_text(token).detach() for token in text_token.split(80)]
        txt_feats = txt_feats[0] if len(txt_feats) == 1 else torch.cat(txt_feats, dim=0)
        
        del model
        torch.cuda.empty_cache()

        if use_cache:
            if not os.path.exists(wv_dir):
                os.makedirs(wv_dir)
            torch.save(txt_feats, cache_file)

        return txt_feats

### ORIGINAL CODE ###
### DEPRECATED ###

def obj_edge_vectors_glove(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0,1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word
            lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
            # print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]

    return vectors

def rel_vectors_glove(names, wv_dir, wv_type='glove.6B', wv_dim=300):
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_dir, wv_type, wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)  # 51, 200
    vectors.normal_(0, 1)
    for i, token in enumerate(names):
        if i == 0:
            continue
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            split_token = token.split(' ')
            ss = 0
            s_vec = torch.zeros(wv_dim)
            for s_token in split_token:
                wv_index = wv_dict.get(s_token)
                if wv_index is not None:
                    ss += 1
                    s_vec += wv_arr[wv_index]
                else:
                    print("fail on {}".format(token))
            s_vec /= ss
            vectors[i] = s_vec

    return vectors

def load_word_vectors(root, wv_type, dim):
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    URL = {
        'glove.42B': 'http://nlp.stanford.edu/data/glove.42B.300d.zip',
        'glove.840B': 'http://nlp.stanford.edu/data/glove.840B.300d.zip',
        'glove.twitter.27B': 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        'glove.6B': 'http://nlp.stanford.edu/data/glove.6B.zip',
        }
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)

    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('loading word vectors from', fname_pt)
        try:
            return torch.load(fname_pt, map_location=torch.device("cpu"))
        except Exception as e:
            print("Error loading the model from {}{}".format(fname_pt, str(e)))
            sys.exit(-1)
    else:
        print("INFO File not found: ", fname + '.pt')
    if not os.path.isfile(fname + '.txt'):
        print("INFO File not found: ", fname + '.txt')
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    elif os.path.basename(wv_type) in URL:
        url = URL[wv_type]
        print('downloading word vectors from {}'.format(url))
        filename = os.path.basename(fname)
        if not os.path.exists(root):
            os.makedirs(root)
        with tqdm(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
            fname, _ = urlretrieve(url, fname, reporthook=reporthook(t))
            with zipfile.ZipFile(fname, "r") as zf:
                print('extracting word vectors into {}'.format(root))
                zf.extractall(root)
        if not os.path.isfile(fname + '.txt'):
            raise RuntimeError('no word vectors of requested dimension found')
        return load_word_vectors(root, wv_type, dim)
    else:
        raise RuntimeError('unable to load word vectors')

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret

def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]
    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner
