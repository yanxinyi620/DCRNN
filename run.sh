# ------------------------ Data Preparation
# Create data directories
mkdir -p data/{METR-LA,PEMS-BAY}

# METR-LA
python -m scripts.generate_training_data --output_dir=data/METR-LA --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python -m scripts.generate_training_data --output_dir=data/PEMS-BAY --traffic_df_filename=data/pems-bay.h5


# ------------------------ Graph Construction
python -m scripts.gen_adj_mx  --sensor_ids_filename=data/sensor_graph/graph_sensor_ids.txt --normalized_k=0.1\
    --output_pkl_filename=data/sensor_graph/adj_mx.pkl


# ------------------------ Model Training
# METR-LA
# python dcrnn_train.py --config_filename=data/model/dcrnn_la.yaml
python dcrnn_train.py

# PEMS-BAY
python dcrnn_train.py --config_filename=data/model/dcrnn_bay.yaml


# ------------------------ Run the Pre-trained Model on METR-LA
# METR-LA
# python run_demo.py --config_filename=data/model/pretrained/METR-LA/config.yaml
python run_demo.py

# PEMS-BAY
python run_demo.py --config_filename=data/model/pretrained/PEMS-BAY/config.yaml


# ------------------------ Eval baseline methods
# METR-LA
python -m scripts.eval_baseline_methods --traffic_reading_filename=data/metr-la.h5


# ------------------------ GPU status
nvidia-smi

