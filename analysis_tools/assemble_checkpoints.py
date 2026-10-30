from pathlib import Path
import argparse,hashlib,json,urllib.request
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');args=p.parse_args()
metadata=json.loads((root/'CHECKPOINT_PARTS.json').read_text())
output=root/metadata['archive']
assert not output.exists(), 'Output archive already exists; inspect it before rebuilding'
for item in metadata['parts']:
 file=root/item['name']
 if not file.exists() and args.download:urllib.request.urlretrieve(item['url'],file)
 assert file.exists(), 'Missing '+str(file)+'; use --download or provide all parts'
 assert file.stat().st_size==item['bytes'] and hashlib.sha256(file.read_bytes()).hexdigest()==item['sha256'],item['name']
with output.open('wb') as target:
 for item in metadata['parts']:
  with (root/item['name']).open('rb') as source:
   while block:=source.read(1024*1024):target.write(block)
assert output.stat().st_size==metadata['bytes'] and hashlib.sha256(output.read_bytes()).hexdigest()==metadata['sha256']
print('Verified archive:',output)
print('Extract it at the repository root to restore the original checkpoint paths.')
