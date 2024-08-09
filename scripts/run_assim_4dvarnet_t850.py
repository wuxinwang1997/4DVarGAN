import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py model=fdvarnet paths=assim_openi_t850 datamodule=ncassimilate_t850 datamodule.batch_size=256 model.pred_ckpt=/tmp/dataset/ckpt_t850/afnonet_t850.ckpt trainer.max_epochs=50 test=True  && \    
    echo "train done" 
    """)