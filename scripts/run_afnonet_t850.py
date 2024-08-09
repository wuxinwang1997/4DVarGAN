import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py model=afnonet paths=forecast_openi_t850 datamodule=ncforecast_t850 test=True  && \    
    echo "train done" 
    """)