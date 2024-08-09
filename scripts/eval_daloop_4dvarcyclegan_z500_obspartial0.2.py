import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=2880 --model_name=4dvarunet_woscale --obs_partial=0.2 && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=2880 --model_name=4dvarunet_wscale --obs_partial=0.2 && \
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=2880 --model_name=4dvargan_woscale --obs_partial=0.2 && \    
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=2880 --model_name=4dvargan_wscale --obs_partial=0.2 && \         
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=2880 --model_name=4dvarcyclegan_woscale --obs_partial=0.2 && \         
    python src/evaluate/eval_daloop.py --cycle_hours=720 --decorrelation_hours=2880 --model_name=4dvarcyclegan_wscale --obs_partial=0.2 && \         
    echo "train done" 
    """)