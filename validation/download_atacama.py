import asf_search as asf, json, os, glob
from datetime import datetime
sel=set(json.load(open('data/atacama_sel.json')))
have={os.path.basename(f).rsplit('.nc',1)[0] for f in glob.glob('data/atacama_gunw/*.nc')}
need=sel-have
print(f"{len(have)} ya, {len(need)} por bajar", flush=True)
res=asf.search(intersectsWith='POINT(-68.25 -23.55)', dataset='ARIA S1 GUNW',
               relativeOrbit=156, flightDirection='DESCENDING', start='2019-09-01', end='2020-04-05')
sess=asf.ASFSession()
todo=[r for r in res if r.properties['sceneName'] in need]
for i,r in enumerate(todo,1):
    try:
        r.download(path='data/atacama_gunw', session=sess)
        print(f"[{i}/{len(todo)}] {r.properties['sceneName'][:46]}", flush=True)
    except Exception as e:
        print("ERR", r.properties['sceneName'][:40], e, flush=True)
print("DONE", flush=True)
