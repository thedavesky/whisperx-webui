# Ported from https://github.com/openai/whisper/blob/main/whisper/utils.py

import json
import os
import math
import re
import sys
import zlib
from typing import Callable, List, Optional, TextIO, Union, Dict, Tuple
from datetime import datetime

import gradio as gr

from modules.whisper.data_classes import Segment, Word
from .files_manager import read_file

LANGUAGES_WITHOUT_SPACES = {"ja", "zh"}


def format_timestamp(
    seconds: float, always_include_hours: bool = True, decimal_marker: str = ","
) -> str:
    assert seconds is not None and seconds >= 0, "Wrong timestamp provided"

    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def time_str_to_seconds(time_str: str, decimal_marker: str = ",") -> float:
    times = time_str.split(":")

    if len(times) == 3:
        hours, minutes, rest = times
        hours = int(hours)
    else:
        hours = 0
        minutes, rest = times

    seconds, fractional = rest.split(decimal_marker)

    minutes = int(minutes)
    seconds = int(seconds)
    fractional_seconds = float("0." + fractional)

    return hours * 3600 + minutes * 60 + seconds + fractional_seconds


def get_start(segments: List[dict]) -> Optional[float]:
    return next(
        (w["start"] for s in segments for w in (s.get("words") or [])),
        segments[0]["start"] if segments else None,
    )


def get_end(segments: List[dict]) -> Optional[float]:
    return next(
        (w["end"] for s in reversed(segments) for w in reversed(s.get("words") or [])),
        segments[-1]["end"] if segments else None,
    )


def read_subtitle_lines(file_path: str) -> List[str]:
    """Read a subtitle file as lines, tolerating a UTF-8 BOM and CRLF endings."""
    content = read_file(file_path).lstrip("\ufeff").replace("\r\n", "\n")
    return content.split("\n")


def parse_cue_block(block: List[str], decimal_marker: str, file_path: str, context: str,
                    require_timing: bool = True) -> Optional[Segment]:
    """Build one Segment from a SRT/VTT cue block.

    The cue identifier (SRT index, VTT cue id) is optional, the end timestamp may
    carry VTT cue settings, and multiline dialogue is kept verbatim. A block
    without dialogue text is skipped; a block with a broken timestamp fails with
    the file, block and line in the message. `require_timing` is False for WebVTT,
    whose header, NOTE and LIBRARY blocks legitimately carry no cue timing.
    """
    lines = [line.strip() for line in block]
    timing_pos = next((i for i, line in enumerate(lines) if " --> " in line), None)
    if timing_pos is None:
        if require_timing:
            raise gr.Error(f"{file_path}: {context} has no timestamp line: {lines[0]!r}")
        return None

    time_line = lines[timing_pos]
    start_str, end_str = time_line.split(" --> ", 1)
    try:
        start = time_str_to_seconds(start_str.strip(), decimal_marker)
        # VTT cue settings (`position:10% align:start`) follow the end timestamp.
        end = time_str_to_seconds(end_str.split()[0], decimal_marker)
    except Exception as e:
        raise gr.Error(
            f"{file_path}: invalid timestamp in {context} ({time_line!r}): {e}"
        )

    text = "\n".join(lines[timing_pos + 1:]).strip()
    if not text:
        return None
    return Segment(start=start, end=end, text=text)


LRC_STAMP = re.compile(r"\[(\d[^\]]*:[^\]]*)\]")


def require_timed_cue(extension: str, start: Optional[str], end: Optional[str]) -> None:
    """Reject cues a format cannot express (e.g. a start-only LRC line written as SRT)."""
    if start is None or end is None:
        missing = "only a start" if end is None else "only an end"
        raise gr.Error(
            f"{extension.upper()} needs a start and an end timestamp for every cue, "
            f"but one cue has {missing}. Use LRC for lines without an end timestamp."
        )


def split_cue_blocks(lines: List[str]) -> List[Tuple[List[str], int]]:
    """Split SRT/VTT content into cue blocks, keeping each block's first line number."""
    blocks: List[Tuple[List[str], int]] = []
    block: List[str] = []
    start_line = 0
    for lineno, line in enumerate(lines, start=1):
        if line.strip() == "":
            if block:
                blocks.append((block, start_line))
                block = []
        else:
            if not block:
                start_line = lineno
            block.append(line)
    if block:
        blocks.append((block, start_line))
    return blocks


def parse_lrc_line(line: str, decimal_marker: str, file_path: str, lineno: int) -> List[Segment]:
    """Turn one LRC line into segments.

    Handles the conventional `[mm:ss.xx]lyric` form, the same lyric repeated under
    several timestamps, and this app's word-timed form
    `[mm:ss.xxx]word[mm:ss.xxx]word[mm:ss.xxx]`, where a trailing stamp is the end
    of the last word. A line without a closing stamp keeps `end=None`; the end is
    only inferred from the next cue when the line itself supplies that stamp.
    """
    tokens = LRC_STAMP.split(line)
    stamps = tokens[1::2]
    texts = tokens[2::2]
    if not stamps:
        return []

    segments: List[Segment] = []
    pending: List[float] = []
    for index, stamp in enumerate(stamps):
        text = texts[index] if index < len(texts) else ""
        try:
            seconds = time_str_to_seconds(stamp, decimal_marker)
        except Exception as e:
            raise gr.Error(
                f"{file_path}: invalid LRC timestamp [{stamp}] on line {lineno}: {e}"
            )
        if not text.strip():
            pending.append(seconds)
            continue
        # Every stamp that was waiting for this text gets the same lyric.
        for start in pending:
            segments.append(Segment(start=start, end=None, text=text.strip()))
        pending = []
        segments.append(Segment(start=seconds, end=None, text=text.strip()))

    if pending and segments:
        # A stamp after the final text closes that cue, so the word timings chain up.
        for index, segment in enumerate(segments[:-1]):
            segment.end = segments[index + 1].start
        segments[-1].end = pending[0]
    return segments


def extract_probability(word: Union[Word, dict, None]) -> Optional[float]:
    if word is None:
        return None
    if isinstance(word, Word):
        return word.probability
    if isinstance(word, dict):
        return word.get("probability")
    return None


def get_segment_confidence(segment: Union[Segment, dict]) -> Optional[float]:
    if isinstance(segment, Segment):
        segment_dict = segment.model_dump()
    else:
        segment_dict = segment

    word_confidences = [
        prob
        for prob in (
            extract_probability(word)
            for word in segment_dict.get("words") or []
        )
        if prob is not None
    ]
    if word_confidences:
        return sum(word_confidences) / len(word_confidences)

    avg_logprob = segment_dict.get("avg_logprob")
    if avg_logprob is not None:
        return math.exp(avg_logprob)
    return None


def format_confidence(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    return f"{value:.2f}"


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(
        self, result: Union[dict, List[Segment]], output_file_name: str,
            options: Optional[dict] = None, **kwargs
    ):
        if isinstance(result, List) and all(isinstance(seg, Segment) for seg in result):
            result = {"segments": [seg.model_dump() for seg in result]}

        output_path = os.path.join(
            self.output_dir, output_file_name + "." + self.extension
        )

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f, options=options, **kwargs)

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        raise NotImplementedError

    def to_segments(self, file_path: str):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(
        self, result: Union[Dict, List[Segment]], file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for segment in result["segments"]:
            if segment["text"] is None:
                continue
            print(segment["text"].strip(), file=file, flush=True)

    def to_segments(self, file_path: str):
        segments = []

        for line in read_subtitle_lines(file_path):
            if line.strip() != '':
                segments.append(Segment(
                    start=None,
                    end=None,
                    text=line.strip()
                ))
        return segments


class SubtitlesWriter(ResultWriter):
    always_include_hours: bool
    decimal_marker: str

    def iterate_result(
        self,
        result: dict,
        options: Optional[dict] = None,
        *,
        max_line_width: Optional[int] = None,
        max_line_count: Optional[int] = None,
        highlight_words: bool = False,
        align_lrc_words: bool = False,
        max_words_per_line: Optional[int] = None,
    ):
        options = options or {}
        max_line_width = max_line_width or options.get("max_line_width")
        max_line_count = max_line_count or options.get("max_line_count")
        highlight_words = highlight_words or options.get("highlight_words", False)
        align_lrc_words = align_lrc_words or options.get("align_lrc_words", False)
        max_words_per_line = max_words_per_line or options.get("max_words_per_line")
        include_confidence = options.get("include_confidence", False)
        preserve_segments = max_line_count is None or max_line_width is None
        max_line_width = max_line_width or 1000
        max_words_per_line = max_words_per_line or 1000

        language = (result.get("language") or "").lower()
        spaced_language = language not in LANGUAGES_WITHOUT_SPACES

        def render_segment_confidence(segment: dict) -> str:
            confidence = format_confidence(get_segment_confidence(segment)) if include_confidence else None
            return f" (conf: {confidence})" if confidence is not None else ""

        def render_word_label(word: dict) -> str:
            label = word.get("word", "")
            speaker = word.get("speaker")
            if speaker:
                label = f"[{speaker}] {label.strip()}"
            if include_confidence:
                word_confidence = format_confidence(extract_probability(word))
                if word_confidence is not None:
                    label = f"{label} ({word_confidence})"
            return label

        def segment_speaker(segment: dict) -> Optional[str]:
            speaker = segment.get("speaker")
            text = segment.get("text") or ""
            if not speaker and "|" in text:
                candidate = text.split("|", 1)[0].strip()
                if candidate.upper().startswith("SPEAKER"):
                    speaker = candidate
            return speaker

        def strip_speaker_prefix(text: str, speaker: Optional[str]) -> str:
            if speaker and text.strip().startswith(f"{speaker}|"):
                return text.strip()[len(speaker) + 1:].strip()
            return text.strip()

        def usable_words(words: List[dict]) -> bool:
            """Word timing is usable only when every word has finite, ordered bounds."""
            previous_start = None
            for word in words:
                start, end = word.get("start"), word.get("end")
                if start is None or end is None:
                    return False
                if not (math.isfinite(start) and math.isfinite(end)):
                    return False
                if end < start:
                    return False
                if previous_start is not None and start < previous_start:
                    return False
                previous_start = start
            return True

        def spaced_words(segment: dict) -> List[dict]:
            """Restore the spacing/punctuation WhisperX alignment strips from words.

            The words are located in the segment text in order, so the original
            inter-word characters (spaces, punctuation) are reused verbatim; no
            space is invented for languages that are written without spaces.
            """
            text = strip_speaker_prefix(segment.get("text") or "", segment_speaker(segment))
            # `List[Segment]` results carry no language, so detect unspaced scripts here.
            # shortcut: CJK ranges only; if the writer gains the language, use it directly.
            spaced = spaced_language and not re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text)
            cursor = 0
            rendered: List[dict] = []
            for word in segment["words"]:
                stripped = (word.get("word") or "").strip()
                if not stripped:
                    continue
                found = text.find(stripped, cursor)
                if found >= 0:
                    label = text[cursor:found] + stripped
                    cursor = found + len(stripped)
                elif spaced:
                    label = (" " if cursor > 0 else "") + stripped
                else:
                    label = stripped
                rendered.append({**word, "word": label})
            return rendered

        def segment_level_cues(segment: dict):
            text = segment.get("text")
            if text is None:
                return
            speaker = segment_speaker(segment)
            text = strip_speaker_prefix(text, speaker).replace("-->", "->")
            confidence = render_segment_confidence(segment)
            if confidence:
                text = f"{text}{confidence}"
            yield (
                self.format_timestamp(segment["start"]) if segment.get("start") is not None else None,
                self.format_timestamp(segment["end"]) if segment.get("end") is not None else None,
                text,
            )

        def word_chunks(segment_words: List[dict], last: float):
            line_len = 0
            line_count = 1
            subtitle: List[dict] = []
            for original_timing in segment_words:
                timing = original_timing.copy()
                long_pause = not preserve_segments and timing["start"] - last > 3.0
                has_room = line_len + len(timing["word"]) <= max_line_width
                if line_len > 0 and has_room and not long_pause:
                    # line continuation
                    line_len += len(timing["word"])
                else:
                    # new line
                    timing["word"] = timing["word"].strip()
                    if (
                        len(subtitle) > 0
                        and max_line_count is not None
                        and (long_pause or line_count >= max_line_count)
                    ):
                        # subtitle break
                        yield subtitle
                        subtitle = []
                        line_count = 1
                    elif line_len > 0:
                        # line break
                        line_count += 1
                        timing["word"] = "\n" + timing["word"]
                    line_len = len(timing["word"].strip())
                subtitle.append(timing)
                last = timing["start"]
                if len(subtitle) >= max_words_per_line:
                    yield subtitle
                    subtitle = []
                    line_len, line_count = 0, 1
            if len(subtitle) > 0:
                yield subtitle

        last: float = get_start(result["segments"]) or 0.0
        for segment in result["segments"]:
            words = segment.get("words")
            if not words or not usable_words(words):
                yield from segment_level_cues(segment)
                continue

            speaker = segment_speaker(segment)
            first_chunk = True
            for subtitle in word_chunks(spaced_words(segment), last):
                last = subtitle[-1]["end"]
                subtitle_start = self.format_timestamp(subtitle[0]["start"])
                subtitle_end = self.format_timestamp(subtitle[-1]["end"])
                subtitle_text = "".join([render_word_label(word) for word in subtitle])
                if first_chunk and speaker:
                    subtitle_text = f"[{speaker}] {subtitle_text}"
                first_chunk = False

                if align_lrc_words:
                    lrc_aligned_words = [
                        f"[{self.format_timestamp(word['start'])}]{render_word_label(word)}"
                        for word in subtitle
                    ]
                    last_word = subtitle[-1]
                    lrc_aligned_words[-1] = (
                        f"[{self.format_timestamp(last_word['start'])}]"
                        f"{render_word_label(last_word)}"
                        f"[{self.format_timestamp(last_word['end'])}]"
                    )
                    # The word labels already carry the original inter-word spaces.
                    yield None, None, "".join(lrc_aligned_words)

                elif highlight_words:
                    # Per-word karaoke cues only: no overlapping whole-segment duplicate.
                    all_words = [render_word_label(word) for word in subtitle]
                    for i, word_timing in enumerate(subtitle):
                        word_start = self.format_timestamp(word_timing["start"])
                        word_end = self.format_timestamp(word_timing["end"])
                        yield word_start, word_end, "".join(
                            [
                                re.sub(r"^(\s*)(.*)$", r"\1<u>\2</u>", label)
                                if j == i
                                else label
                                for j, label in enumerate(all_words)
                            ]
                        )

                else:
                    yield subtitle_start, subtitle_end, subtitle_text

    def format_timestamp(self, seconds: float):
        return format_timestamp(
            seconds=seconds,
            always_include_hours=self.always_include_hours,
            decimal_marker=self.decimal_marker,
        )


class WriteVTT(SubtitlesWriter):
    extension: str = "vtt"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("WEBVTT\n", file=file)
        for start, end, text in self.iterate_result(result, options, **kwargs):
            require_timed_cue(self.extension, start, end)
            print(f"{start} --> {end}\n{text}\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        segments = []

        for block, start_line in split_cue_blocks(read_subtitle_lines(file_path)):
            segment = parse_cue_block(block, self.decimal_marker, file_path,
                                      f"cue at line {start_line}", require_timing=False)
            if segment is not None:
                segments.append(segment)

        return segments


class WriteSRT(SubtitlesWriter):
    extension: str = "srt"
    always_include_hours: bool = True
    decimal_marker: str = ","

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        for i, (start, end, text) in enumerate(
            self.iterate_result(result, options, **kwargs), start=1
        ):
            require_timed_cue(self.extension, start, end)
            print(f"{i}\n{start} --> {end}\n{text}\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        segments = []

        for block, start_line in split_cue_blocks(read_subtitle_lines(file_path)):
            segment = parse_cue_block(block, self.decimal_marker, file_path,
                                      f"cue at line {start_line}")
            if segment is not None:
                segments.append(segment)

        return segments


class WriteLRC(SubtitlesWriter):
    extension: str = "lrc"
    always_include_hours: bool = False
    decimal_marker: str = "."

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        options = {**(options or {}), **kwargs}
        align_words = options.get("align_lrc_words", False)
        for start, end, text in self.iterate_result(result, options):
            if align_words:
                print(f"{text}\n", file=file, flush=True)
            elif end is None:
                # A conventional lyric line has a start stamp and no end stamp.
                print(f"[{start}]{text}\n", file=file, flush=True)
            else:
                print(f"[{start}]{text}[{end}]\n", file=file, flush=True)

    def to_segments(self, file_path: str) -> List[Segment]:
        segments = []

        for lineno, line in enumerate(read_subtitle_lines(file_path), start=1):
            if line.strip() != '':
                segments.extend(parse_lrc_line(line.strip(), self.decimal_marker, file_path, lineno))

        return segments


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            if segment["start"] is None or segment["end"] is None:
                raise gr.Error(
                    "TSV needs a start and an end timestamp for every segment, but the "
                    "input is untimed text (for example a .txt file). Use TXT, LRC, SRT or VTT."
                )
            print(round(1000 * segment["start"]), file=file, end="\t")
            print(round(1000 * segment["end"]), file=file, end="\t")
            print(segment["text"].strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(
        self, result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
    ):
        json.dump(result, file)


def get_writer(
    output_format: str, output_dir: str
) -> Callable[[dict, TextIO, dict], None]:
    output_format = output_format.strip().lower().replace(".", "")

    writers = {
        "txt": WriteTXT,
        "vtt": WriteVTT,
        "srt": WriteSRT,
        "tsv": WriteTSV,
        "json": WriteJSON,
        "lrc": WriteLRC
    }

    if output_format == "all":
        all_writers = [writer(output_dir) for writer in writers.values()]

        def write_all(
            result: dict, file: TextIO, options: Optional[dict] = None, **kwargs
        ):
            for writer in all_writers:
                writer(result, file, options, **kwargs)

        return write_all

    return writers[output_format](output_dir)


def generate_file(
    output_format: str,
    output_dir: str,
    result: Union[dict, List[Segment]],
    output_file_name: str,
    add_timestamp: bool = True,
    options: Optional[dict] = None,
    **kwargs,
) -> Tuple[str, str]:
    output_format = output_format.strip().lower().replace(".", "")
    output_format = "vtt" if output_format == "webvtt" else output_format

    if add_timestamp:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_file_name += f"-{timestamp}"

    file_path = os.path.join(output_dir, f"{output_file_name}.{output_format}")
    file_writer = get_writer(output_format=output_format, output_dir=output_dir)

    # Writer options arrive both as a dict and as keyword arguments; normalize once.
    writer_options = {**(options or {}), **kwargs}
    if output_format == "lrc" and writer_options.get("highlight_words"):
        writer_options["highlight_words"], writer_options["align_lrc_words"] = False, True

    file_writer(
        result=result,
        output_file_name=output_file_name,
        options=writer_options,
    )
    content = read_file(file_path)
    return content, file_path


def safe_filename(name):
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    MAX_FILENAME_LENGTH = 200
    safe_name = re.sub(INVALID_FILENAME_CHARS, '_', name)
    # Truncate the filename if it exceeds the max_length (MAX_FILENAME_LENGTH)
    if len(safe_name) > MAX_FILENAME_LENGTH:
        file_extension = safe_name.split('.')[-1]
        if len(file_extension) + 1 < MAX_FILENAME_LENGTH:
            truncated_name = safe_name[:MAX_FILENAME_LENGTH - len(file_extension) - 1]
            safe_name = truncated_name + '.' + file_extension
        else:
            safe_name = safe_name[:MAX_FILENAME_LENGTH]
    return safe_name
