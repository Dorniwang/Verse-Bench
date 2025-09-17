from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class SenseVoiceInferencer:
    def __init__(self, model_path):
        model_dir = f"{model_path}/SenseVoiceSmall"

        # 如果自己下载的sensevoice模型，建议删除目录下的requirements.txt文件，否则你不知道那帮老登会给你的环境里装什么。
        model = AutoModel(
            model=model_dir,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device="cuda:0",
            hub="hf",
            disable_update=True
        )
        self.model = model

    def infer(self, audio_path):
        res = self.model.generate(
            input=audio_path,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,  #
            merge_length_s=15,
        )
        text = rich_transcription_postprocess(res[0]["text"])
        return text

