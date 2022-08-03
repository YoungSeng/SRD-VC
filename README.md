# Speech Representation Disentanglement with Adversarial Mutual Information Learning for One-shot Voice Conversion

## 1. Demo Page

The demo  can be found [here](https://im1eon.github.io/IS2022-SRDVC/)

## 2. Dependencies

Install required python packages:

`pip install -r requirements.txt`

## 3. Preparation, Training and Inference

Download the VCTK dataset.

Extract spectrogram and f0:`make_spect_f0.py`. 

And modify it to your own path and divide the dataset, run `data_split.py`.

Generate training metadata: `make_metadata.py`.

Run the training scripts: `main.py`.

Generate testing metadata: `make_test_metadata.py`.

Run the inference scripts: `inference.py`

## 4. Evaluation

You may refer to the following: `WER.py`, `mcd.py`, `f0_pcc.py`, `draw_f0_distributions.py`, `draw_speaker_embedding.py`

## 5. References

Our work mainly inspired by:

(1) [SpeechSplit](https://github.com/auspicious3000/SpeechSplit#readme):
```
K. Qian, Y. Zhang, S. Chang, M. Hasegawa-Johnson, and D. Cox, “Unsupervised speech decomposition via triple information bottleneck,” in International Conference on Machine Learning. PMLR, 2020, pp. 7836–7846.
```

(2) [VQMIVC](https://github.com/Wendison/VQMIVC):
```
D. Wang, L. Deng, Y. T. Yeung, X. Chen, X. Liu, and H. Meng, “VQMIVC: Vector Quantization and Mutual Information-Based Unsupervised Speech Representation Disentanglement for OneShot Voice Conversion,” in Interspeech, 2021, pp. 1344–1348.
```

(3) ClsVC:
```
H. Tang, X. Zhang, J. Wang, N. Cheng, and J. Xiao, “Clsvc: Learning speech representations with two different classification tasks.” Openreview, 2021, https://openreview.net/forum?id=xp2D-1PtLc5.
```

[comment]: <> (## 5. Citation)

[comment]: <> (If you find our work useful in your research, please consider citing:)

[comment]: <> (```)

[comment]: <> (@inproceedings{yang2022Speech,)

[comment]: <> (  author={SiCheng Yang and Methawee Tantrawenith and Haolin Zhuang and Zhiyong Wu and Aolan Sun and Jianzong Wang and ning cheng and Huaizhen Tang and Xintao Zhao and Jie Wang and Helen Meng},)

[comment]: <> (  title={{Speech Representation Disentanglement with Adversarial Mutual Information Learning for One-shot Voice Conversion}},)

[comment]: <> (  year=2022,)

[comment]: <> (  booktitle={Proc. Interspeech 2022},)

[comment]: <> (})

[comment]: <> (```)
[comment]: <> (  pages={846--850},)

[comment]: <> (  doi={10.21437/Interspeech.2021-1990})