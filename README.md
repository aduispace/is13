**Note:** I don't provide personal support for custom changes in the code. Only
for the release.  For people just starting, I recommend
[Treehouse](http://referrals.trhou.se/grgoiremesnil) for online-learning.

Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding
==============================================================================================================

### Code for RNN and Spoken Language Understanding

Based on the Interspeech '13 paper:

[Grégoire Mesnil, Xiaodong He, Li Deng and Yoshua Bengio - **Investigation of Recurrent Neural Network Architectures and Learning Methods for Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/RNNSpokenLanguage2013.pdf)

We also have a follow-up IEEE paper:

[Grégoire Mesnil, Yann Dauphin, Kaisheng Yao, Yoshua Bengio, Li Deng, Dilek Hakkani-Tur, Xiaodong He, Larry Heck, Gokhan Tur, Dong Yu and Geoffrey Zweig - **Using Recurrent Neural Networks for Slot Filling in Spoken Language Understanding**](http://www.iro.umontreal.ca/~lisa/pointeurs/taslp_RNNSLU_final_doubleColumn.pdf)

## Code

This code allows to get state-of-the-art results and a significant improvement
(+1% in F1-score) with respect to the results presented in the paper.

Run the following commands:

```
git clone https://github.com/Liang-Qiu/is13.git
cd is13
virtualenv venv -p python3
source venv/bin/activate
pip3 install -r requirements-gpu.txt
python examples/elman-keras.py
```

## ATIS Data

Download ATIS Dataset here! [split 0](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold0.pkl.gz) [split 1](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold1.pkl.gz) [split 2](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold2.pkl.gz) [split 3](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold3.pkl.gz) [split 4](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/mesnilgr/atis/atis.fold4.pkl.gz)

```
import _pickle as cPickle
train, valid, test, dicts = cPickle.load(gzip.open('atis.fold0.pkl.gz', 'rb'), encoding='latin1')
```

`dicts` is a python dictionnary that contains the mapping from the labels, the
name entities (if existing) and the words to indexes used in `train` and `test`
lists. Refer to this [tutorial](http://deeplearning.net/tutorial/rnnslu.html) for more details. 

Running the following command can give you an idea of how the data has been preprocessed:

```
python data/load.py
```

## License

<a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" property="dct:title">Recurrent Neural Network Architectures for Spoken Language Understanding</span> by <span xmlns:cc="http://creativecommons.org/ns#" property="cc:attributionName">Grégoire Mesnil</span> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc/4.0/">Creative Commons Attribution-NonCommercial 4.0 International License</a>.<br />Based on a work at <a xmlns:dct="http://purl.org/dc/terms/" href="https://github.com/mesnilgr/is13" rel="dct:source">https://github.com/mesnilgr/is13</a>.
