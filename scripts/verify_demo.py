"""Small integrity check for exported evidence and selected media; no GPU work."""
import hashlib
import json
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

ROOT=Path(__file__).resolve().parents[1]
prov=json.loads((ROOT/'results/provenance.json').read_text())
for name,entry in prov['files'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==entry['sha256'],name
media=json.loads((ROOT/'demo/media-manifest.json').read_text())
for entry in media:
    p=ROOT/entry['path']
    assert p.stat().st_size==entry['bytes'],str(p)
    assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],str(p)
class Links(HTMLParser):
    def __init__(self):super().__init__();self.paths=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ('href','src','poster') and v:self.paths.append(v)
for file in [ROOT/'demo/index.html',ROOT/'demo/study.html']:
    parser=Links();parser.feed(file.read_text())
    for link in parser.paths:
        url=urlsplit(link)
        if not url.scheme and url.path:
            assert (file.parent/unquote(url.path)).is_file(),(file,link)
data=json.loads((ROOT/'demo/assets/data.json').read_text())
for case in data['cases']:
    for g in case['records']:
        for ending in ['.mp4','-start.png','-trace.npz']:
            assert (ROOT/f'demo/media/{case["id"]}/{g}{ending}').is_file()
conversion=json.loads((ROOT/'results/dataset/conversion.json').read_text())
assert hashlib.sha256((ROOT/'results/dataset/norm_stats.json').read_bytes()).hexdigest()==conversion['norm_stats_sha256']
assert conversion['train_frames']==61750 and conversion['train_episodes']==500
assert conversion['validation_frames']==0 and conversion['validation_episodes']==0
assert len(list((ROOT/'demo/assets/charts').glob('*.png')))==9
assert len(list((ROOT/'demo/assets/charts').glob('*.svg')))==9
print(f'PASS: {len(prov["files"])} exported files unchanged; {len(media)} selected media files verified; data protocol and local HTML links valid.')
