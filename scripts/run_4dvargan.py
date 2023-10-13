import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    python src/train.py --config-name=train_da datamodule=z500_predda_multiobs datamodule.random_erase=false datamodule.pred_len=1 model=fdvargan trainer.max_epochs=50 test=fasle && \
    echo "train done" 
    """)