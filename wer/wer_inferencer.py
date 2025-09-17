import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import scipy
import string
from wer.sensevoice_inferencer import SenseVoiceInferencer


class WERInferencer:
    def __init__(self, model_path):
        self.device = "cuda"
        self.model = SenseVoiceInferencer(f"{model_path}")

    def get_asr(self, audio_path):
        return self.model.infer(audio_path)

    def infer(self, audio_path1, audio_path2):
        asr1 = self.get_asr(audio_path1)
        asr2 = self.get_asr(audio_path2)
        measures = jiwer.wer(asr1, asr2)
        return measures

    def infer_audio_text(self, audio_path, text):
        asr = self.get_asr(audio_path)
        text = text.strip().lower()
        PUNCTUATION_SET = set(string.punctuation)
        if set(asr).issubset(PUNCTUATION_SET):
            asr = ""
        asr = asr.strip().lower()
        measures = jiwer.wer(text, asr)
        return measures

    def infer_text_text(self, text1, text2):
        text1 = text1.strip().lower()
        text2 = text2.strip().lower()
        measures = jiwer.wer(text1, text2)
        return measures

