#!/usr/bin/env python3
"""Figuras chilenas para el showcase (estilo editorial: papel, tinta, acento)."""
import json, os, datetime as dt
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK="#24303a"; WARM="#b5451f"; COOL="#2f6f86"
plt.rcParams.update({"font.family":"DejaVu Sans","text.color":INK,"axes.edgecolor":INK,
  "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.linewidth":0.8,
  "savefig.transparent":True,"axes.facecolor":"none","figure.dpi":150})
FIG="/tmp/insar_chile_figs"; os.makedirs(FIG, exist_ok=True)

def load(d):
    m=json.load(open(f"validation/{d}/meta.json")); nr,nc=m["rows"],m["cols"]; g=m["geo"]
    rd=lambda f: np.fromfile(f"validation/{d}/{f}",np.float32).reshape(nr,nc)
    return m,nr,nc,g,rd
def ext(g,nr,nc): return [g["lon0"],g["lon0"]+nc*g["dlon"],g["lat0"]+nr*g["dlat"],g["lat0"]]
def nomap(ax): ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]

# ---- Maule: velocidad + serie temporal ----
m,nr,nc,g,rd=load("maule_summer_export")
vel=rd("velocity.f32")*100; tc=rd("tcoh.f32"); ser=np.fromfile("validation/maule_summer_export/series.f32",np.float32).reshape(m["n_epochs"],nr,nc)
vm=np.where(tc>0.7,vel,np.nan)
py,px=np.unravel_index(np.nanargmax(np.where(tc>0.8,np.abs(vel),np.nan)),vel.shape)
ep=[dt.datetime.strptime(s,"%Y-%m-%d") for s in m["epochs"]]; days=np.array([(d-ep[0]).days for d in ep]); yr=days/365.25
disp=(ser[:,py,px]-ser[:,4,702])*100; v,b=np.polyfit(yr,disp,1)
vmx=np.nanpercentile(np.abs(vm),99)
fig,ax=plt.subplots(figsize=(5.2,4.4))
im=ax.imshow(vm,cmap="RdBu_r",vmin=-vmx,vmax=vmx,extent=ext(g,nr,nc))
ax.plot(g["lon0"]+px*g["dlon"],g["lat0"]+py*g["dlat"],"o",mfc="none",mec=INK,mew=1.8,ms=13)
nomap(ax); cb=fig.colorbar(im,ax=ax,shrink=.82,pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
cb.set_label("velocidad LOS (cm/año)",fontsize=9)
fig.tight_layout(pad=.3); fig.savefig(f"{FIG}/maule_vel.png",bbox_inches="tight"); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.4,3.3))
ax.axhline(0,color=INK,lw=.5,alpha=.3)
ax.plot(days,disp,"o",ms=4.5,color=WARM,mec="white",mew=.5,zorder=3)
ax.plot(days,v*yr+b,"-",color=INK,lw=1.4,label=f"{v:.1f} cm/año")
ax.set_xlabel(f"días desde {m['epochs'][0]}",fontsize=9); ax.set_ylabel("desplazamiento LOS (cm)",fontsize=9)
ax.tick_params(labelsize=8); [ax.spines[s].set_visible(False) for s in ("top","right")]
ax.legend(frameon=False,fontsize=10,loc="best")
fig.tight_layout(pad=.4); fig.savefig(f"{FIG}/maule_ts.svg",bbox_inches="tight"); plt.close(fig)
maule_v=round(float(v),1); maule_disp=round(float(abs(disp[-1])),0)

# ---- Atacama: velocidad ----
m2,nr2,nc2,g2,rd2=load("atacama_export")
av=rd2("velocity.f32")*100; at=rd2("tcoh.f32"); avm=np.where(at>0.7,av,np.nan)
fig,ax=plt.subplots(figsize=(5.2,4.4))
im=ax.imshow(avm,cmap="RdBu_r",vmin=-6,vmax=6,extent=ext(g2,nr2,nc2))
nomap(ax); cb=fig.colorbar(im,ax=ax,shrink=.82,pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=8)
cb.set_label("velocidad LOS (cm/año)",fontsize=9)
fig.tight_layout(pad=.3); fig.savefig(f"{FIG}/atacama_vel.png",bbox_inches="tight"); plt.close(fig)

# ---- Cadena: gap-fill (geostat) + exposición (ABM) ----
filled=rd("velocity_filled.f32"); exp=rd("exposure.f32")
fig,ax=plt.subplots(1,2,figsize=(10,4.2))
im=ax[0].imshow(np.where(np.isfinite(filled),filled,np.nan),cmap="RdBu_r",vmin=-vmx,vmax=vmx,extent=ext(g,nr,nc))
ax[0].set_title("campo continuo — kriging (geostat-rs)",fontsize=10); nomap(ax[0]); plt.colorbar(im,ax=ax[0],shrink=.8).outline.set_visible(False)
ev=np.where(exp>0,exp,np.nan)
im=ax[1].imshow(ev,cmap="inferno",extent=ext(g,nr,nc))
ax[1].set_title("exposición — ABM (swarm-abm)",fontsize=10); nomap(ax[1]); plt.colorbar(im,ax=ax[1],shrink=.8).outline.set_visible(False)
fig.tight_layout(pad=.4); fig.savefig(f"{FIG}/chain.png",bbox_inches="tight"); plt.close(fig)

json.dump({"maule_v":maule_v,"maule_disp":maule_disp}, open(f"{FIG}/stats.json","w"))
print("figuras chilenas →", FIG, "| Maule v=", maule_v, "cm/año, disp=", maule_disp, "cm")
