import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=24 && \ 
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=48 && \         
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=72 && \         
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=96 && \                 
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=168 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 --init_time=120 && \                 
    echo "train done" 
    """)