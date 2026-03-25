CSV_PATH = '/home/kuchoco97/work/2026_01_age_predict/csv/final_data.csv'
CSV_PATH_NORMAL = '/home/kuchoco97/work/2026_01_age_predict/csv/final_normal.csv'
FEATURE_ROOT = '/data0/age_predict/preprocessed/coords/features_uni_v2'

# caculated from GTEx data
AGE_MEAN = 53.830
AGE_STD = 12.489

# Tissue maaping
# 1. 서브타입을 큰 범주의 장기(Organ) 이름으로 매핑하는 딕셔너리
SUBTYPE_TO_ORGAN = {
    'Adipose_Subcutaneous': 'Adipose',
    'Adipose_Visceral_Omentum': 'Adipose',
    'Adrenal_Gland': 'Adrenal_Gland',
    'Artery_Aorta': 'Artery',
    'Artery_Coronary': 'Artery',
    'Artery_Tibial': 'Artery',
    'Bladder': 'Bladder',
    'Brain_Cerebellum': 'Brain',
    'Brain_Cortex': 'Brain',
    'Breast_Mammary_Tissue': 'Breast',
    'Cervix_Ectocervix': 'Cervix',
    'Cervix_Endocervix': 'Cervix',
    'Colon_Sigmoid': 'Colon',
    'Colon_Transverse': 'Colon',
    'Esophagus_Gastroesophageal_Junction': 'Esophagus',
    'Esophagus_Mucosa': 'Esophagus',
    'Esophagus_Muscularis': 'Esophagus',
    'Fallopian_Tube': 'Fallopian_Tube',
    'Heart_Atrial_Appendage': 'Heart',
    'Heart_Left_Ventricle': 'Heart',
    'Kidney_Cortex': 'Kidney',
    'Kidney_Medulla': 'Kidney',
    'Liver': 'Liver',
    'Lung': 'Lung',
    'Minor_Salivary_Gland': 'Minor_Salivary_Gland',
    'Muscle_Skeletal': 'Muscle',
    'Nerve_Tibial': 'Nerve',
    'Ovary': 'Ovary',
    'Pancreas': 'Pancreas',
    'Pituitary': 'Pituitary',
    'Prostate': 'Prostate',
    'Skin_Not_Sun_Exposed_Suprapubic': 'Skin',
    'Skin_Sun_Exposed_Lower_leg': 'Skin',
    'Small_Intestine_Terminal_Ileum': 'Small_Intestine',
    'Spleen': 'Spleen',
    'Stomach': 'Stomach',
    'Testis': 'Testis',
    'Thyroid': 'Thyroid',
    'Uterus': 'Uterus',
    'Vagina': 'Vagina'
}

# 2. 통합된 29개의 장기(Organ)에 새로운 고유 ID를 부여하는 딕셔너리
ORGAN_TO_ID = {
    'Adipose': 0,
    'Adrenal_Gland': 1,
    'Artery': 2,
    'Bladder': 3,
    'Brain': 4,
    'Breast': 5,
    'Cervix': 6,
    'Colon': 7,
    'Esophagus': 8,
    'Fallopian_Tube': 9,
    'Heart': 10,
    'Kidney': 11,
    'Liver': 12,
    'Lung': 13,
    'Minor_Salivary_Gland': 14,
    'Muscle': 15,
    'Nerve': 16,
    'Ovary': 17,
    'Pancreas': 18,
    'Pituitary': 19,
    'Prostate': 20,
    'Skin': 21,
    'Small_Intestine': 22,
    'Spleen': 23,
    'Stomach': 24,
    'Testis': 25,
    'Thyroid': 26,
    'Uterus': 27,
    'Vagina': 28
}