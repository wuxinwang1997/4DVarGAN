import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py model=afnonet paths=forecast_openi_z500 datamodule=ncforecast_z500 test=True  && \    
    echo "train done" 
    """)