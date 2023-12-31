import os
import pickle
import xmltodict


# https://github.com/unicamp-dl/PTT5/blob/cee2d996ba7eac80d7764072eef01a7f9c38836c/assin/assin_dataset.py
def preprocess_assin2(pickled_file, version=2, subset="ptpt"):
    curDir = os.getcwd() + "/evaluation/assin"
    if version == 2:
        dataDir = os.path.join(curDir, "assin2")
        filenames = ['assin2-train-only.xml', 'assin2-dev.xml', 'assin2-test.xml']
    else:
        dataDir = os.path.join(curDir, "assin1")
        filenames = ['assin-ptpt-train.xml', 'assin-ptpt-dev.xml', 'assin-ptpt-test.xml']
    pickled_file = os.path.join(dataDir, pickled_file)
    print(pickled_file)
    splits = ["train", "validation", "test"]

    if not os.path.isfile(pickled_file):
        processed_data = {mode: [] for mode in splits}

        for split, fname in zip(splits, filenames):
            with open(os.path.join(dataDir, fname), 'r') as xml:
                xml_dict = xmltodict.parse(xml.read())
                for data in xml_dict['entailment-corpus']['pair']:
                    processed_data[split].append({
                        "sentence_pair_id": data['@id'],
                        "premise": data['t'],
                        "hypothesis": data['h'],
                        "relatedness_score": float(data['@similarity']),
                        "entailment_judgment": entailment_id(data['@entailment'])})
        with open(pickled_file, 'wb') as processed_file:
            pickle.dump(processed_data, processed_file)
    else:
        with open(pickled_file, 'rb') as processed_file:
            processed_data = pickle.load(processed_file)
    return processed_data


def entailment_id(entailment):
    if entailment == "Entailment":
        return 1
    return 0
