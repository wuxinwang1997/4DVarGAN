import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=vit --obs_partial=0.1 && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarnet --obs_partial=0.1 && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.1 && \         
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvar --inflation=0.5 --maxIter=1 --obs_partial=0.1 && \
    echo "train done" 
    """)