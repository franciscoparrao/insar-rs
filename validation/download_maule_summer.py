import asf_search as asf, json, os, glob
sel=set(json.load(open('data/maule_summer_sel.json')))
have={os.path.basename(f).rsplit('.nc',1)[0] for f in glob.glob('data/maule_summer_gunw/*.nc')}
need=sel-have
print(f"{len(have)} ya, {len(need)} por bajar", flush=True)
res=asf.search(intersectsWith='POINT(-70.50 -36.06)', dataset='ARIA S1 GUNW',
               relativeOrbit=83, flightDirection='DESCENDING', start='2017-11-15', end='2020-04-15')
sess=asf.ASFSession()
todo=[r for r in res if r.properties['sceneName'] in need]
for i,r in enumerate(todo,1):
    try:
        r.download(path='data/maule_summer_gunw', session=sess)
        print(f"[{i}/{len(todo)}] ok", flush=True)
    except Exception as e:
        print("ERR", e, flush=True)
print("DONE", flush=True)
