import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py --config-name=train_da model=vit_base datamodule=z500_da trainer.max_epochs=50 callbacks.model_checkpoint.monitor='val/mse' callbacks.model_checkpoint.mode='min' callbacks.early_stopping.monitor='val/mse' callbacks.early_stopping.mode='min' test=False  && \    
    echo "train done" 
    """)