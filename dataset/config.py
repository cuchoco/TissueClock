CSV_PATH = '/home/kuchoco97/work/2026_01_age_predict/csv/final_data.csv'
FEATURE_ROOT = '/data0/age_predict/preprocessed/coords/features_uni_v2'

# caculated from GTEx data
AGE_MEAN = 53.830
AGE_STD = 12.489

# Tissue maaping
TISSUE_TO_ID = {
    'Adipose_Subcutaneous': 0,
    'Adipose_Visceral_Omentum': 1,
    'Adrenal_Gland': 2,
    'Artery_Aorta': 3,
    'Artery_Coronary': 4,
    'Artery_Tibial': 5,
    'Bladder': 6,
    'Brain_Cerebellum': 7,
    'Brain_Cortex': 8,
    'Breast_Mammary_Tissue': 9,
    'Cervix_Ectocervix': 10,
    'Cervix_Endocervix': 11,
    'Colon_Sigmoid': 12,
    'Colon_Transverse': 13,
    'Esophagus_Gastroesophageal_Junction': 14,
    'Esophagus_Mucosa': 15,
    'Esophagus_Muscularis': 16,
    'Fallopian_Tube': 17,
    'Heart_Atrial_Appendage': 18,
    'Heart_Left_Ventricle': 19,
    'Kidney_Cortex': 20,
    'Kidney_Medulla': 21,
    'Liver': 22,
    'Lung': 23,
    'Minor_Salivary_Gland': 24,
    'Muscle_Skeletal': 25,
    'Nerve_Tibial': 26,
    'Ovary': 27,
    'Pancreas': 28,
    'Pituitary': 29,
    'Prostate': 30,
    'Skin_Not_Sun_Exposed_Suprapubic': 31,
    'Skin_Sun_Exposed_Lower_leg': 32,
    'Small_Intestine_Terminal_Ileum': 33,
    'Spleen': 34,
    'Stomach': 35,
    'Testis': 36,
    'Thyroid': 37,
    'Uterus': 38,
    'Vagina': 39
}