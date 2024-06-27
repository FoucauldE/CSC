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