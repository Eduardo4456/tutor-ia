EMB_MODEL      = "bge-m3"
LLM_MODEL      = "tutor-estagio"
VISION_MODEL   = "moondream"            # modelo leve de captioning, menos restritivo
CHUNK_SIZE, OVERLAP = 1500, 150
MIN_CHUNK      = 120                # descarta chunks de texto muito curtos
MIN_IMG_PIXELS = 100 * 100          # ignora ícones/decorações (< 10 000 px)
TOP_K          = 4
CSV_PATH             = "data/base_vetorial.csv"
SIMILARITY_THRESHOLD = 0.40            # nota mínima de similaridade para considerar um trecho relevante
 