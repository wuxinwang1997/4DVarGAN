import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py model=cyclegan paths=assim_openi_z500 datamodule=ncassimilate datamodule.batch_size=32 trainer.max_epochs=50 test=True  && \    
    echo "train done" 
    """)