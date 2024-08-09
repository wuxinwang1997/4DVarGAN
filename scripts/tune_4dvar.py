import os

os.system(""" ls /home && echo "show code" && ls /tmp/code && echo "show dataset" && ls /tmp/dataset && echo "show output" && ls /tmp/output  && \
    pip install -r requirements.txt && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.0 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.1 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.2 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.3 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.4 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.5 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.6 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.7 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.8 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=0.9 --maxIter=1 && \
    python src/evaluate/tune_4dvar.py --cycle_hours=720 --init_time=72 --decorrelation_hours=2880 --model_name=4dvar --inflation=1.0 --maxIter=1 && \
    echo "train done" 
    """)