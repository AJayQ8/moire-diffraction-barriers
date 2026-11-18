"""Rearrange archived profiles and model scores for the v021 manuscript.

No fits are recomputed. The same DS10/DS14 cuts and matched-map scores used
in v020 supply the overview, model-error panels and retained outlier figure.
"""
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_mos2_matched_wall_models import map_points, MODELS, GROUPS, plot_rigid_comparison
from plot_measured_wall_examples import BLUE, ORANGE, INK, GREY

PAPER=Path(__file__).resolve().parents[2]
FIGURES=PAPER/'manuscript/figures'
EVIDENCE=PAPER/'evidence'


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
        'axes.labelsize':8,'axes.titlesize':8.5,'xtick.labelsize':7.2,
        'ytick.labelsize':7.2,'legend.fontsize':7.2,'axes.linewidth':.7,
        'lines.linewidth':1.45,'pdf.fonttype':42,'ps.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})


def panel(ax,label,image=False):
    ax.text(.025,.97,f'({label})',transform=ax.transAxes,ha='left',va='top',
        fontsize=8.5,fontweight='bold',color='white' if image else INK,
        bbox={'facecolor':'black' if image else 'white','alpha':.72,
              'edgecolor':'none','pad':1.4},zorder=20)


def image_panel(fig,ax,scan,name,label):
    r=scan['first_ring_image']; pixel=scan['pixel_size_nm']
    a,b,c,d=r['crop_bounds_rc_px_half_open']; origin=np.array(r['display_origin_xy_nm'])
    extent=((c-.5)*pixel-origin[0],(d-.5)*pixel-origin[0],
            (b-.5)*pixel-origin[1],(a-.5)*pixel-origin[1])
    lo,hi=r['display_quantiles_raw_counts']
    im=ax.imshow(np.array(r['raw_integrated_count_crop']),origin='upper',extent=extent,
        cmap='gray_r',vmin=lo,vmax=hi,interpolation='nearest',aspect='equal')
    chord=np.array(r['chord_xy_nm'])-origin
    cut=np.array(r['midpoint_cut_xy_nm'])-origin
    inds=np.array(r['midpoint_cut_displayed_indices'],dtype=int)
    dist=np.array(r['midpoint_cut_distance_nm_full'])
    assert np.array_equal(inds,np.flatnonzero((dist>=scan['profile_distance_nm'][0]-1e-9)&(dist<=scan['profile_distance_nm'][-1]+1e-9)))
    ax.plot(chord[:,0],chord[:,1],color='#16a085',lw=1.6,ls='--',zorder=5)
    ax.scatter(chord[:,0],chord[:,1],s=12,color='#16a085',edgecolors='white',linewidths=.6,zorder=7)
    ax.plot(cut[inds,0],cut[inds,1],color='#c23b32',lw=1.2,zorder=6)
    ax.set(xlim=extent[:2],ylim=extent[2:],xlabel='Image x (nm)',ylabel='Image y (nm)')
    ax.set_title(f"{name}: {scan['row_spacing_nm']:.2f} nm wall spacing",loc='left',pad=5)
    bar=fig.colorbar(im,ax=ax,fraction=.047,pad=.022)
    bar.set_label('Integrated counts',fontsize=7,labelpad=3)
    bar.ax.tick_params(labelsize=6.8,length=2,pad=1)
    panel(ax,label,True)


def profile_panel(ax,scan,key,label):
    x=np.array(scan['profile_distance_nm']);r=scan['profile_channels'][key]
    curves=[np.array(r['plotted_measured_unit_l2_shape'])]
    ax.plot(x,curves[0],color=INK,lw=.75,marker='o',ms=2.5,markevery=2,
        mfc='white',mew=.7,zorder=4)
    for name,color,ls in [('periodic_anchor',BLUE,'-'),('rigid',ORANGE,'--')]:
        y=np.array(scan['model_curves'][name]['unit_l2_shape_by_ring'][key]);curves.append(y)
        ax.plot(x,y,color=color,ls=ls)
    order=1 if key=='first' else 2; hk=r['representative_mode_hk']
    ax.set_title(f"{'First' if order==1 else 'Second'} order ({hk[0]:g}, {hk[1]:g})",loc='left',pad=5)
    ax.set(xlim=(x.min(),x.max()),xlabel='Distance across wall (nm)',ylabel='Normalized profile')
    ax.axhline(0,color=GREY,lw=.55,zorder=0)
    width=scan['measured_widths_nm'][key]
    ax.text(.97,.965,rf'$w_{order}={width:.2f}\,\mathrm{{nm}}$',transform=ax.transAxes,
        ha='right',va='top',fontsize=7.5,bbox={'facecolor':'white','edgecolor':'#aaa','lw':.35,'pad':1.3})
    panel(ax,label)
    return max(float(np.max(np.abs(y))) for y in curves)


def legend(fig):
    handles=[
      Line2D([0],[0],color=INK,lw=.75,marker='o',ms=2.5,mfc='white',mew=.7,label='Measured profile'),
      Line2D([0],[0],color='#16a085',lw=1.6,ls='--',marker='o',ms=3,mec='white',mew=.6,label='Wall segment'),
      Line2D([0],[0],color=BLUE,label='Periodic network'),
      Line2D([0],[0],color='#c23b32',label='Transverse profile'),
      Line2D([0],[0],color=ORANGE,ls='--',label='Rigid layers')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=3,
        frameon=False,handlelength=1.8,columnspacing=1.2)


def gap_panel(ax,points):
    for row in points:
        ax.plot([row['spacing'],row['spacing']+.12],[row['gap']['measured'],row['gap']['network']],
            color='#b9c8d3',lw=.85,zorder=1)
    for name,label,color,marker,shift in MODELS:
        ax.plot([r['spacing']+shift for r in points],[r['gap'][name] for r in points],
            ls='none',marker=marker,ms=5,color=color,mfc='white' if name=='network' else color,
            mew=1.05,label=label,zorder=3)
    ax.axhline(0,color='#aab3b9',lw=.7,zorder=0)
    ax.set(xlabel='Spacing of parallel wall rows (nm)',
        ylabel=r'Mean fitted-width difference $w_1-w_2$ (nm)',
        xlim=(6.5,32.8),ylim=(-.9,2.65),xticks=[8,12,16,20,24,28,32])
    ax.set_title('Width difference across ten maps',loc='left',pad=5)
    ax.legend(loc='upper right',frameon=False,fontsize=6.9,handlelength=1.,
              handletextpad=.35,labelspacing=.25,borderpad=.1)
    panel(ax,'d')


def main():
    style()
    paths=[EVIDENCE/n for n in ('mos2_native_wall_examples_v01.json','mos2_matched_wall_models_v01.json','mos2_matched_wall_models_summary_v01.json','mos2_rigid_width_gaps_v01.json')]
    examples,cases,summary,rigid=[json.loads(p.read_text()) for p in paths]
    assert sha256(paths[1].read_bytes()).hexdigest()==summary['case_sha256']==rigid['case_sha256']
    points=map_points(cases,summary)
    # Figure 1: measured DS10 image and profiles, then all-map spacing trend.
    fig,axes=plt.subplots(2,2,figsize=(7.15,5.50))
    scan=examples['scans']['DS10']
    image_panel(fig,axes[0,0],scan,'DS10','a')
    m1=profile_panel(axes[0,1],scan,'first','b')
    m2=profile_panel(axes[1,0],scan,'second','c')
    for ax in (axes[0,1],axes[1,0]):ax.set_ylim(-1.24*max(m1,m2),1.24*max(m1,m2))
    gap_panel(axes[1,1],points);legend(fig)
    fig.subplots_adjust(left=.085,right=.985,bottom=.10,top=.86,wspace=.41,hspace=.54)
    fig.savefig(FIGURES/'interface_overview_v021.pdf',facecolor='white')
    plt.close(fig)
    # Figure 2: retain the full two-model cohort and the qualified rigid subset.
    fig,axes=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
    ax=axes[0]
    for i,(group,_) in enumerate(GROUPS):
        r=summary['groups'][group]['scan_grid']
        for model,color,offset in [('isolated',ORANGE,-.16),('network',BLUE,.16)]:
            v=r['map_mean_gap_absolute_error_nm'][model]
            ax.bar(i+offset,v,width=.26,color=color,zorder=2,label=('Isolated wall' if model=='isolated' else 'Periodic network') if i==0 else None)
            ax.text(i+offset,v+.04,f'{v:.2f}',color=color,ha='center',va='bottom',fontsize=8)
    ax.set(xlim=(-.55,1.55),ylim=(0,1.65),ylabel='Mean absolute error (nm)',
        xticks=[0,1],xticklabels=['Closer spacing\n168 segments, 6 maps','Wider spacing\n74 segments, 4 maps'],yticks=[0,.4,.8,1.2,1.6])
    ax.set_title('(a) All matched wall segments',loc='left')
    ax.grid(axis='y',color='#dbe0e4',lw=.5,zorder=0)
    ax.legend(loc='upper right',frameon=False,fontsize=7)
    plot_rigid_comparison(axes[1],rigid['groups']['higher_twist'])
    axes[1].set_title('(b) Common subset including rigid layers',loc='left')
    axes[1].set_xlabel('112 segments in 5 maps')
    fig.savefig(FIGURES/'model_errors_v021.pdf',facecolor='white');plt.close(fig)
    # Figure S6: unchanged DS14 outlier, now placed beside the local-residual discussion.
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.82),gridspec_kw={'width_ratios':[1.03,1,1]})
    scan=examples['scans']['DS14'];image_panel(fig,axes[0],scan,'DS14','a')
    m1=profile_panel(axes[1],scan,'first','b');m2=profile_panel(axes[2],scan,'second','c')
    for ax in axes[1:]:ax.set_ylim(-1.24*max(m1,m2),1.24*max(m1,m2))
    legend(fig)
    fig.subplots_adjust(left=.075,right=.985,bottom=.19,top=.70,wspace=.68)
    fig.savefig(FIGURES/'local_width_outlier_v021.pdf',facecolor='white');plt.close(fig)
    receipt={'operation':'Figure rearrangement and relabelling only; no fit or physical calculation.',
        'inputs_sha256':{p.name:sha256(p.read_bytes()).hexdigest() for p in paths},
        'main_example':'DS10 (1,7), unchanged from v020',
        'supplement_example':'DS14 (3,7), unchanged from v020',
        'map_points':points,
        'two_model_segments':{k:summary['groups'][k]['scan_grid']['matched_sections'] for k,_ in GROUPS},
        'rigid_subset_segments':rigid['groups']['higher_twist']['three_model_sections']}
    (PAPER/'results/figure_inputs_v021.json').write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
