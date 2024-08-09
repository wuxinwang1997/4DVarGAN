import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py model=fdvargan_unet paths=assim_openi_z500 datamodule=ncassimilate_z500 datamodule.batch_size=256 trainer.accumulate_grad_batches=1 trainer.max_epochs=50 test=True  && \    
    echo "train done" 
    """)