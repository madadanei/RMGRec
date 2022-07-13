# RMGRec
Core codes of RMGRec here
# Requirements
- System: Ubuntu 20.04
- Devices: a single GTX 3090 GPU
- python 3.8
- pytorch 1.11
# Dataset
- NYC and TKY [description:https://sites.google.com/site/yangdingqi/home/foursquare-dataset; download:http://www-public.it-sudparis.eu/~zhang_da/pub/dataset_tsmc2014.zip]
<br>Dingqi Yang, Daqing Zhang, Vincent W. Zheng, Zhiyong Yu. Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs. IEEE Trans. on Systems, Man, and Cybernetics: Systems, (TSMC), 45(1), 129-142, 2015.
- Gowalla [description:http://snap.stanford.edu/data/loc-gowalla.html; download:http://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz]
<br>E. Cho, S. A. Myers, J. Leskovec. Friendship and Mobility: Friendship and Mobility: User Movement in Location-Based Social Networks ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 2011.
- Put the raw data into directory `data`, then run `data/preprocess.py`. The generatd `session.pickle` and `transition.pickle` are used for RMGRec.
