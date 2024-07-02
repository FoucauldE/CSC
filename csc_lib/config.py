from sklearn.svm import SVC
from xgboost import XGBClassifier

DATA_PATH = "DATASET/corpus_propre"
OUTPUT_PATH = "Outputs"
FT_MODEL_PATH = "../../model/bloom-1b1_lora"

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


GEN_ARGS = {
    "do_sample":  True,
    "num_beams": 1,
    "num_return_sequences": 1,
    "top_k": 40,
    "max_new_tokens": 200,
    "return_dict_in_generate": True,
    "output_scores": True,
}


models_and_params = {
    
    'SVC': {
        'model': SVC(probability=True, random_state=42),
        'param_grid': {
            'model__C': [0.1, 1, 10],
            'model__gamma': [0.001, 0.01, 0.1]
        }
    },

    'xgb': {
        'model': XGBClassifier(eval_metric='logloss', random_state=42),
        'param_grid': {
            'model__n_estimators': [100, 200, 300],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.01, 0.1, 0.3],
            'model__subsample': [0.6, 0.8, 1.0]
        }
    }
}