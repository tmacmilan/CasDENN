# CasDENN
Code for "Eﬀicient and Eﬀective Information Cascade Prediction via Degree Distribution Evolution Neural Networks"
This is the data and code for paper submitted, titled 'Degree Distribution Evolution Neural Networks for Efficient and Effective Information Cascade Prediction'

1:dataset
  (1)APS, raw data is from https://journals.aps.org/datasets, and 'data/APS_preprocessing.txt' is our preprossed data.
  (2)Weibo, it is too large, so we do not update, 'dataset_weibo.txt' can be downloaded from https://github.com/CaoQi92/DeepHawkes
  The format is also the same as in https://github.com/CaoQi92/DeepHawkes
2:config
  config_APS_1.py and config_weibo_1.py are two examples of the config files of these dataset, some settings (N, T, DDI) can be set to get new dataset.

3:Run the model:
  (1) python gen_shortestpath.py  
  (2) python preprocess_graph_signal_degree.py  
  (3) rnn_model.py is about the vairants of deep learning models CasDENN.
  where, 'import config_APS_1 as config' can import new config files.
