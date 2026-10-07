"""KWS 触发后的 ASR 文本复核(两阶段唤醒)。

开放词表 KWS 对生僻专名唤醒词(如「列克星敦」)误触率偏高:
KWS 只做第一阶段粗检,命中后把触发点前的音频送离线 ASR 识别,
与配置的唤醒词做模糊匹配,匹配不过则丢弃本次触发。
环形缓冲由 KWS 检测线程驱动(push 与 verify 同线程,无需加锁),
ASR 识别与连续对话共用 SherpaASR 单例(onnxruntime session 线程安全)。
"""

import time
from collections import deque
from difflib import SequenceMatcher

from core.services.audio.asr.sherpa import SherpaASR
from core.utils.config import ConfigManager
from core.utils.logger import logger


class KwsAsrVerifier:
    """KWS 命中后的 ASR 复核器。"""

    def __init__(self, frame_size: int, sample_rate: int):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.enabled = False
        self.threshold = 0.55
        self.keywords: list = []
        self._buffer: deque = deque()
        self._max_frames = 0
        self.apply_runtime_config()

    def apply_runtime_config(self):
        """同步最新复核配置(配置热重载时由 KWS 一并触发)。"""
        cfg = ConfigManager.instance()
        self.enabled = bool(cfg.get_app_config("kws.verify_by_asr", False))
        window_ms = float(cfg.get_app_config("kws.verify_window_ms", 2500))
        self.threshold = float(cfg.get_app_config("kws.verify_threshold", 0.55))
        self.keywords = [
            keyword
            for keyword in cfg.get_app_config("wakeup.keywords", [])
            if isinstance(keyword, str) and keyword
        ]
        self._max_frames = max(
            1, int(window_ms * self.sample_rate / 1000 / self.frame_size)
        )

    def push(self, frames: bytes):
        """把检测线程读到的每帧音频存入环形缓冲(无论 VAD/KWS 是否激活)。"""
        if self._max_frames <= 0:
            return
        self._buffer.append(frames)
        while len(self._buffer) > self._max_frames:
            self._buffer.popleft()

    def verify(self, kws_text: str) -> bool:
        """复核触发点前的音频是否真的说了唤醒词。

        复核开关关闭或无可用信息时放行(fail-open);
        ASR 异常时放行并告警,避免识别故障拖垮唤醒。
        """
        if not self.enabled or not self.keywords:
            return True
        pcm = b"".join(self._buffer)
        if not pcm:
            return True

        start = time.time()
        try:
            text = SherpaASR.asr(pcm)
        except Exception as exc:
            logger.warning(
                f"[KWS-Verify] ASR 复核异常,放行触发: {type(exc).__name__}: {exc}",
                module="KWS",
            )
            return True

        if not text:
            logger.info(
                f"[KWS-Verify] kws={kws_text} asr='' 复核未识别到语音,丢弃触发",
                module="KWS",
            )
            return False

        ratio = max(
            SequenceMatcher(None, text, keyword).ratio()
            for keyword in self.keywords
        )
        passed = ratio >= self.threshold
        elapsed_ms = (time.time() - start) * 1000
        logger.info(
            f"[KWS-Verify] kws={kws_text} asr={text!r} "
            f"ratio={ratio:.2f}/{self.threshold} "
            f"-> {'放行' if passed else '丢弃'} ({elapsed_ms:.0f}ms)",
            module="KWS",
        )
        return passed
