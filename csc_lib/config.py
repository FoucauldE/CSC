import os

DATA_PATH = "DATASET/corpus_propre"
ANN_PATH = "DATASET/ann_majoritaires"
OUTPUT_PATH = "Outputs"

GEN_PATH = os.path.join(DATA_PATH, 'generated_test_anns.json')
TRAIN_PATH = os.path.join(DATA_PATH, 'train.json')
VAL_PATH = os.path.join(DATA_PATH, 'val.json')

TYPES_TO_KEEP = {'PROC', 'DISO', 'CHEM'}

FILTER_OUT = {
    'examen clinique',
    'examen',
    'traitement',
    'hospitalisation',
    'consulté',
    'admission',
    'signes',
    'consulté',
    'consultation',
    'prise en charge',
    'suivi',
    'suivie',
    'hospitalisé',
    'recherche',
    'admis',
    'admise',
    'hospitalisé',
    'bilan',
    'traitée',
    'consulte',
    'maladie',
    'geste',
    'adressé',
    'observation',
    'Elle',
    'elle',
    'il',
    'Il',
    'consultait',
    'processus'
    }