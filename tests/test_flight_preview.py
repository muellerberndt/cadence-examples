"""The preview stays unlisted and its large archive has a checked boundary."""
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_pages import sha256, unpack_recording


def test_flight_is_not_an_official_example_link():
    for relative in ['README.md', 'index.html', 'hub_published.html', 'shared/gallery.html', 'shared/shell.html']:
        assert 'previews/1943' not in (ROOT/relative).read_text()
    html = (ROOT/'previews/1943/index.html').read_text()
    assert 'name="robots" content="noindex, nofollow"' in html
    assert 'href="/' not in html and 'src="/' not in html


@pytest.mark.parametrize('name,kind', [('../escape.json', tarfile.REGTYPE),
                                     ('recording/link.json', tarfile.SYMTYPE),
                                     ('index.html', tarfile.REGTYPE)])
def test_archive_cannot_escape_or_overwrite_the_viewer(tmp_path, name, kind):
    archive = tmp_path/'unsafe.tar'
    with tarfile.open(archive, 'w') as tar:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = '/etc/passwd' if kind == tarfile.SYMTYPE else ''
        member.size = 0
        tar.addfile(member, io.BytesIO())
    spec = {'sha256': sha256(archive), 'bytes': archive.stat().st_size, 'files': 1}
    with pytest.raises(ValueError, match='Unsafe or unexpected'):
        unpack_recording(archive, tmp_path/'site', spec)
    assert not (tmp_path/'escape.json').exists()


def test_recording_is_pinned_and_complete():
    spec = json.loads((ROOT/'previews/1943/recording.json').read_text())
    assert spec['frames'] == 528 and spec['iterations'] == 10870
    assert len(spec['sha256']) == 64
    assert spec['bytes'] < 950_000_000
