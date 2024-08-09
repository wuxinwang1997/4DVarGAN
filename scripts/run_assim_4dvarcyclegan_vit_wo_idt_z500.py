import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py model=fdvarcyclegan_vit_wo_idt paths=assim_openi_z500 datamodule=ncassimilate_z500 datamodule.batch_size=128 trainer.accumulate_grad_batches=2 trainer.max_epochs=50 test=False  && \    
    echo "train done" 
    """)