import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/inference/pred_da_loop_inference_interface.py --obs_partial=0.2 --obs_single=False --da_method='4DVarGAN' --ai_ensemble=1 --obserr_level=0.015 --resolution=5.625 --dt_obs=6 --decorrelation_time=2400 --init_time=120 && \
    python src/inference/pred_da_loop_inference_interface.py --obs_partial=0.15 --obs_single=False --da_method='4DVarGAN' --ai_ensemble=1 --obserr_level=0.015 --resolution=5.625 --dt_obs=6 --decorrelation_time=2400 --init_time=120 && \
    python src/inference/pred_da_loop_inference_interface.py --obs_partial=0.1 --obs_single=False --da_method='4DVarGAN' --ai_ensemble=1 --obserr_level=0.015 --resolution=5.625 --dt_obs=6 --decorrelation_time=2400 --init_time=120 && \
    python src/inference/pred_da_loop_inference_interface.py --obs_partial=0.05 --obs_single=False --da_method='4DVarGAN' --ai_ensemble=1 --obserr_level=0.015 --resolution=5.625 --dt_obs=6 --decorrelation_time=2400 --init_time=120 && \
    echo "eval 4dvargan done"
    """)