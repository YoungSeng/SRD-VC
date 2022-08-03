import os
import jiwer
from espnet2.bin.asr_inference import Speech2Text
import soundfile

speech2text = Speech2Text.from_pretrained(
    "Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best",
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.3,
    lm_weight=0.5,
    penalty=0.0,
    nbest=1
)

ground_truth = []
hypothesis = []

target_dir = "/ceph/home/yangsc21/Python/autovc/Final/My_model___/dim2_pitch_wo_adv_mi/"
source_dir = "/ceph/datasets/VCTK-Corpus/txt/"
output_file = "/ceph/home/yangsc21/Python/autovc/Final/My_model___/WER_dim2_pitch_wo_adv_mi.txt"

transformation = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.Strip(),      #
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),        #
    jiwer.RemovePunctuation(),      #
    jiwer.ReduceToListOfListOfWords(word_delimiter=" "),
])

transformation_ = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.Strip(),
    jiwer.RemoveWhiteSpace(replace_by_space=True),
    jiwer.RemoveMultipleSpaces(),
    jiwer.ExpandCommonEnglishContractions(),
    jiwer.RemovePunctuation(),
    jiwer.transforms.ReduceToListOfListOfChars(),
])

for item in os.listdir(target_dir):
    [source_speaker, target_speaker, uid, condition] = item.split('_')
    with open(output_file, 'a', encoding='utf-8') as output:
        with open(source_dir + source_speaker + '/' + source_speaker + '_' + uid[:3] + '.txt', 'r', encoding='utf-8') as f:
            ground_truth.append(f.read().strip())
            audio, rate = soundfile.read(target_dir + item)
            nbests = speech2text(audio)
            text, *_ = nbests[0]
            hypothesis.append(text)
            output.write(str(jiwer.wer(ground_truth[-1], hypothesis[-1], truth_transform=transformation,
                                       hypothesis_transform=transformation)) + '\t' + source_speaker + '\t' + target_speaker + '\t' + uid + '\t' + ground_truth[-1] + '\t' + hypothesis[-1] + '\n')
            output.write(str(jiwer.cer(ground_truth[-1], hypothesis[-1], truth_transform=transformation_,
                                       hypothesis_transform=transformation_)) + '\t' + source_speaker + '\t' + target_speaker + '\t' + uid + '\t' +
                         ground_truth[-1] + '\t' + hypothesis[-1] + '\n')

error = jiwer.wer(ground_truth, hypothesis,
                  truth_transform=transformation,
                  hypothesis_transform=transformation
                  )
error_ = jiwer.cer(ground_truth, hypothesis,
                  truth_transform=transformation_,
                  hypothesis_transform=transformation_
                  )

print(error)
print(error_)
