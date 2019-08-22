# Stepwise, Label-based Adversarial Training for Unsupervised Video Summarization

## PyTorch Implementation of SUM-GAN-sl
- From **"A Stepwise, Label-based Approach for Improving the Adversarial Training in Unsupervised Video Summarization"** (to appear in the 1st Int. Workshop on AI for Smart TV Content Production, Access and Delivery (AI4TV '19) at ACM Multimedia (ACM MM) 2019, October 21, 2019, Nice, France)
- Written by Evlampios Apostolidis, Alexandros I. Metsai, Eleni Adamantidou, Vasileios Mezaris and Ioannis Patras
- Based on a PyTorch implementation of a variation of the SUM-GAN model, made by [Jaemin Cho](https://github.com/j-min)
- This software can be used for training a GAN-based deep learning architecture for automatic video summarization. Training is performed in a fully unsupervised manner without the need for ground-truth data (i.e. human-generated video summaries). After being unsupervisingly trained on a collection of videos, the SUM-GAN-sl model is capable of producing representative summaries for unseen videos, according to a user-specified time-budget about the summary duration.

## Main dependencies
- Python  3.6
- PyTorch 1.0.1

## Data
Structured h5 files with the video features and annotations of the SumMe and TVSum datasets are available within the "data" folder. The GoogleNet features of the video frames were extracted by [Ke Zhang](https://github.com/kezhang-cs) and [Wei-Lun Chao](https://github.com/pujols) and the h5 files were obtained from [Kaiyang Zhou](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce). These files have the following structure:
<pre>
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /gtscore                  1D-array with shape (n_steps), stores ground truth improtance score (used for training, e.g. regression loss)
    /user_summary             2D-array with shape (num_users, n_frames), each row is a binary vector (used for test)
    /change_points            2D-array with shape (num_segments, 2), each row stores indices of a segment
    /n_frame_per_seg          1D-array with shape (num_segments), indicates number of frames in each segment
    /n_frames                 number of frames in original video
    /picks                    positions of subsampled frames in original video
    /n_steps                  number of subsampled frames
    /gtsummary                1D-array with shape (n_steps), ground truth summary provided by user (used for training, e.g. maximum likelihood)
    /video_name (optional)    original video name, only available for SumMe dataset
</pre>
Original videos and annotations for each dataset are also available in the authors' project webpages:
- TVSum dataset: https://github.com/yalesong/tvsum
- SumMe dataset: https://gyglim.github.io/me/vsum/index.html#benchmark

## Data Preparation
For N-fold cross validation, create training / testing splits of the data using 'create_split.py'. To run this script, please set the arguments below:
<pre>
--dataset: Path to the h5 file of the dataset.
--save-dir: Path to save the output json file with the created splits.
--save-name: Name to save the output json file, excluding the extension.
--num-splits: Number of splits to generate (5 in our case).
--train-percent: Percentage of data that will be used for training (80% in our case).
</pre>
## Training
For training the model using a single split, run:
<pre>
python main.py --split_index N (with N being the index of the split)
</pre>
Alternatively, to train the model for N splits, use the 'run_splits.sh' script according to the following:
<pre>
chmod +x run_splits.sh    # Makes the script executable.
./run_splits              # Runs the script.  
</pre>
Please note that after each training epoch the algorithm performs an evaluation step, and uses the trained model to compute the importance scores for the frames of each test video. These scores are then used by the provided evaluation scripts to assess the overal performance of the model (in F-Score).

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: tensorboard --logdir=/path/to/log-directory --host=localhost
- opening a browser and pasting the returned link from cmd

## Configurations
Setup for the training process:

- On 'data_loader.py', specify the path to the 'h5' file of the dataset and the path to the 'json' file containing data about the created splits.
- On 'configs.py', define the directory where the models will be saved to.
    
Arguments in 'configs.py': 
<pre>
--video_type: The used dataset for training the model. Can be either 'TVSum' or 'SumMe'.
--input_size: The size of the input feature vectors (1024 for GoogLeNet features).
--hidden_size: The hidden size of the LSTM units.
--num_layers: The number of layers of each LSTM network.
--regularization_factor: The value of the reguralization factor (ranges from 0.0 to 1.0).
--n_epochs: Number of training epochs.
--clip: The gradient clipping parameter.
--lr: Learning rate.
--discriminator_lr: Discriminator's learning rate.
--split_index: The index of the current split.
</pre>
## Evaluation
To evaluate the models using the computed importance scores for each test video and after each training epoch, run the 'check_fscores_summe.py' and 'check_fscores_tvsum.py' scripts, after specifying:
<pre>
a) the path to the folder where the json files with the analysis results (i.e. frame-level importance scores) are stored
b) the path to the h5 files of the datasets.
</pre>
## Citation
If you find this code useful in your work, please cite the following publication:

E. Apostolidis, A. I. Metsai, E. Adamantidou, V. Mezaris, I. Patras. **"A Stepwise, Label-based Approach for Improving the Adversarial Training in Unsupervised Video Summarization".** In the 1st Int. Workshop on AI for Smart TV Content Production, Access and Delivery (AI4TV '19) at ACM Multimedia (ACM MM) 2019, October 21, 2019, Nice, France

## License
Copyright (c) 2019, Evlampios Apostolidis, Alexandros I. Metsai, Eleni Adamantidou, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.

## Acknowledgement
This work was supported by the European Union Horizon 2020 research and innovation programme under contract H2020-780656 ReTV.
