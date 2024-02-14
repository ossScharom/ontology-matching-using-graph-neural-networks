
python src/omunet/main.py train NCIT_TO_DOID -c config/dev.conf &
python src/omunet/main.py train OMIM_TO_ORDO -c config/dev.conf &
python src/omunet/main.py train SNOMED_TO_FMA_BODY -c config/dev.conf &
python src/omunet/main.py train SNOMED_TO_NCIT_NEOPLAS -c config/dev.conf &
python src/omunet/main.py train SNOMED_TO_NCIT_PHARM -c config/dev.conf &
