"""The voice mode for listeners who cannot see the page (22 October 2026): one button with a spoken name in every
language, a live region that says each step, and no verse ever given to a synthetic voice. The page is checked as
text here; the conversation itself was tried in a browser with a simulated microphone."""
import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parent.parent / "muhawir" / "static" / "index.html").read_text(encoding="utf-8")


def test_the_button_and_the_live_region_are_on_the_page():
    assert 'id="voiceMode"' in PAGE and 'id="voiceStatus" role="status" aria-live="assertive"' in PAGE


def test_every_language_names_each_step():
    for key in ("voiceMode", "voiceModeStop", "vmOn", "vmListening", "vmReading", "vmNotHeard", "vmOff", "vmNoMic",
                "vmTranslation"):
        assert len(re.findall(rf"\b{key}: \"", PAGE)) == 3, key  # Arabic, English, Urdu


def test_a_verse_is_heard_only_from_the_reciter():
    body = PAGE[PAGE.index("async function voiceSay"):PAGE.index("async function voiceListen")]
    assert "replace(/﴿[^﴾]*(?:﴾|$)/g, mark)" in body  # quoted verses are left out of synthetic speech
    read = PAGE[PAGE.index("async function voiceRead"):PAGE.index("function startVoiceMode")]
    assert "everyayah.com/data/Alafasy_128kbps" in read and 's.kind !== "quran"' in read
