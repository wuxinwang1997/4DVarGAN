import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=vit --obs_partial=0.2 && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=vit --obs_partial=0.15 && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=vit --obs_partial=0.1 && \         
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=vit --obs_partial=0.05 && \
    echo "train done" 
    """)