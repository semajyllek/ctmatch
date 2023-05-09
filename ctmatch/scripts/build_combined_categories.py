
from typing import Dict, List, Tuple
import json


COMBINED_CAT_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/combined_categories.jsonl'
CAT_SAVE_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/doc_categories.csv'
INDEX2ID_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/index2id.csv'
INDEX2ID_FIX_PATH = '/Users/jameskelly/Documents/cp/ctmatch/data/index2id_fix.csv'




def load_category_dict(cat_path=COMBINED_CAT_PATH) -> Tuple[List, Dict[str, List[float]]]:
    """
    desc:   gets category dict from category path
    """
    sorted_cat_keys = None
    
    with open(cat_path, 'r') as json_file:
        json_list = list(json_file)
            
    all_cat_dict = {}
    for s in json_list:
        s_data = json.loads(s)
        nct_id, cat_dict = s_data.popitem()
        
        if sorted_cat_keys is None:
            sorted_cat_keys = sorted(cat_dict.keys())

        all_cat_dict[nct_id] = [cat_dict[k] for k in sorted_cat_keys]

    return sorted_cat_keys, all_cat_dict



def load_idx2id(index2id_path: str = INDEX2ID_PATH) -> Dict[str, int]:
    """
    desc:   loads id2idx from csv path
    """
    index2id = {}
    with open(index2id_path, 'r') as f:
        for line in f:
            if len(line) < 2:
                continue
            idx, nct_id = line.split(',')
            index2id[idx] = nct_id.strip(' \n')
    
    return index2id



def rebuild_idx2id(index2id_path: str = INDEX2ID_PATH) -> None:
    """
    desc:   loads id2idx from csv path
    """

    index2id = load_idx2id(index2id_path)
    with open(INDEX2ID_FIX_PATH, 'w') as f:
        for i, (idx, nct_id) in enumerate(index2id.items()):
            f.write(f'{i},{nct_id}\n')

       
    


def build_cat_csv(save_path: str = CAT_SAVE_PATH) -> None:
    """
    desc:   builds csv file for category data
            VERY important that the indexes (order) match the order of the embeddings (for nctid lookup in idx2id)
    """
    sorted_cat_keys, cat_dict = load_category_dict()
    idx2id = load_idx2id()
    
    with open(save_path, 'w') as f:
        f.write(','.join(sorted_cat_keys))
        f.write('\n')
        for _, nct_id in idx2id.items():
            cat_vec = cat_dict[nct_id]
            cat_vec_str = ','.join([str(c) for c in cat_vec])
            f.write(cat_vec_str)
            f.write('\n')


if __name__ == '__main__':
    rebuild_idx2id()
    #build_cat_csv()

    
    
    



