import re
from pathlib import Path


def init_project_context():
    """动态导入父模块"""
    import os
    import sys

    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


init_project_context()

from core.utils.config import ConfigManager
from core.utils.file import get_model_file_path
from core.utils.logger import logger


def should_generate_keywords():
    """Return whether keyword generation should run."""
    import os

    xiaozhi_enabled = os.environ.get("XIAOZHI_ENABLE", "").lower() in ("1", "true", "yes")
    # 兼容 OPENCLAW_ENABLE (新) 和 OPENCLAW_ENABLED (旧)
    openclaw_env = os.environ.get("OPENCLAW_ENABLE") or os.environ.get("OPENCLAW_ENABLED") or ""
    openclaw_enabled = openclaw_env.lower() in ("1", "true", "yes")
    openai_enabled = os.environ.get("OPENAI_ENABLE", "").lower() in (
        "1",
        "true",
        "yes",
    )
    qwenpaw_enabled = os.environ.get("QWENPAW_ENABLE", "").lower() in (
        "1",
        "true",
        "yes",
    )

    if (
        not xiaozhi_enabled
        and not openclaw_enabled
        and not openai_enabled
        and not qwenpaw_enabled
    ):
        return False, "XIAOZHI_ENABLE, OPENCLAW_ENABLE/OPENCLAW_ENABLED, OPENAI_ENABLE and QWENPAW_ENABLE are all disabled"

    return True, ""


def get_args():
    config = ConfigManager.instance()
    tokens = get_model_file_path("tokens.txt")
    bpe_model = get_model_file_path("bpe.model")
    output = get_model_file_path("keywords.txt")
    keywords = config.get_app_config("wakeup.keywords", [])
    # 音素版唤醒词行(模型包无 bpe.model 时使用,如 zh-en-3M 系列)
    phonemes = config.get_app_config("wakeup.keywords_phonemes", [])
    texts = [f"{keyword.upper()}" for keyword in keywords]
    return locals()


def _load_token_set(tokens_path: str) -> set:
    """读取 tokens.txt 的 token 集合(每行 "token id" 或 "token")。"""
    tokens = set()
    with open(tokens_path, "r", encoding="utf8") as f:
        for line in f:
            parts = line.split()
            if parts:
                tokens.add(parts[0])
    return tokens


def write_phoneme_keywords(phonemes, tokens_path: str, output: str) -> int:
    """音素模式:逐行校验配置中的音素唤醒词后写入 keywords.txt。

    行格式: "tok tok ... [:boost] [#threshold] [@输出标签]"。
    以 :/#/@ 开头的是 sherpa-onnx 行内特殊标记(:词级加成分数、#词级阈值、
    @输出标签),不参与 token 校验,原样透传;其余 token 必须在 tokens.txt 中。
    """
    token_set = _load_token_set(tokens_path)
    lines = []
    for line in phonemes:
        if not isinstance(line, str) or not line.strip():
            continue
        tokens = line.split()
        body = [
            t
            for t in tokens
            if not t.startswith((":", "#", "@"))
        ]
        unknown = [t for t in body if t not in token_set]
        if unknown:
            logger.error(
                f"Keyword line has unknown tokens {unknown}: {line}",
                module="KWS",
            )
            continue
        lines.append(" ".join(tokens))
    if not lines:
        logger.error(
            "No valid phoneme keyword lines in wakeup.keywords_phonemes",
            module="KWS",
        )
        return 1
    with open(output, "w", encoding="utf8") as f:
        for line in lines:
            f.write(line + "\n")
    logger.debug(f"Keyword file generated (phoneme mode): {output}", module="KWS")
    return 0


def main():
    should_run, reason = should_generate_keywords()
    if not should_run:
        logger.debug(f"Keyword generation skipped: {reason}", module="KWS")
        return 0

    required_files = [Path(get_model_file_path("tokens.txt"))]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        logger.debug(
            "Keyword generation failed: missing model files: "
            f"{', '.join(missing_files)}",
            module="KWS",
        )
        return 1

    args = get_args()

    bpe_path = Path(args["bpe_model"])
    if not bpe_path.is_file():
        # 音素版模型(如 sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20)
        # 不带 bpe.model,cjkchar+bpe 切分不可用,改用配置直供的音素行
        return write_phoneme_keywords(
            args["phonemes"], args["tokens"], args["output"]
        )

    from sherpa_onnx import text2token

    encoded_texts = text2token(
        args["texts"],
        tokens=args["tokens"],
        tokens_type="cjkchar+bpe",
        bpe_model=args["bpe_model"],
    )
    with open(args["output"], "w", encoding="utf8") as f:
        for _, txt in enumerate(encoded_texts):
            line = "".join(txt)
            if re.match(r"^[▁A-Z\s]+$", line):
                f.write(" ".join(txt) + "\n")
            else:
                f.write(" ".join(txt) + f" @{line}" + "\n")
    logger.debug(f"Keyword file generated: {args['output']}", module="KWS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
