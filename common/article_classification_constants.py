from transformers import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING

MAX_SEQ_LEN = 512

SUPPORTED_MODELS = [
    model.__name__.replace("Config", "").lower()
    for model in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys()
]

SUPPORTED_MODELS = sorted([x for y in SUPPORTED_MODELS for x in y])

