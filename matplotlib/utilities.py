#!/usr/bin/env python
"""
matplotlib utilities
"""
from __future__ import print_function, absolute_import, division


import numpy as np
import string as _string
import matplotlib as _matplotlib

import matplotlib.pyplot as plt

def autoscale_y(ax,margin=0.1):
    """
    This function rescales the y-axis based on the data that is visible given the current xlim of the axis.

    Parameters
    ----------
    ax : matplotlib axes object
    margin : int
        the fraction of the total height of the y-data to pad the upper and lower ylims

    Notes
    -----
    taken from https://stackoverflow.com/questions/29461608/matplotlib-fixing-x-axis-scale-and-autoscale-y-axis
    """

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)



alpha_list = list(_string.ascii_lowercase)
Alpha_list = list(_string.ascii_uppercase)


def pretty_subfig_label(ax,
                        tag_index=None,
                        tag_alpha=None,
                        Cap=False,
                        pospad=0,
                        corner=0,
                        fontcolor='white',
                        boxcolor='black',
                        alpha=0.7,
                        lpad=1,
                        rpad=1,
                        fontsize='large',
                        fontweight='bold',
                        zorder=3,
                        edgecolor='black',
                        text_pad=0,
                        xy=None):
    """ make pretty box labels
    
    input:
    ======
    tag_index: integer or list of ints
               integer index for tag {converted to alpha}
    tag_alpha: string or list of strings
               alpha tag to apply to axis
    
    corner: int (default=0)
            {0:upper right, 1:lower right, 2:lower left, 3:upper left}
    """

    ax_all = np.atleast_1d(ax).flatten()
    s = ax_all.shape[0]

    if tag_index is None and tag_alpha is None:
        if Cap:
            tag = Alpha_list[:s]
        else:
            tag = alpha_list[:s]

    if tag_alpha is not None:
        tag = list(np.atleast_1d(tag_alpha))

    if tag_index is not None:
        tag_index_use = np.atleast_1d(tag_index)
        if Cap:
            tag = [Alpha_list[x] for x in tag_index_use]
        else:
            tag = [alpha_list[x] for x in tag_index_use]

    assert s == len(tag)

    if corner == 0:
        x = 1 - pospad
        y = 1 - pospad
        va = 'top'
        ha = 'right'
    elif corner == 1:
        x = 1 - pospad
        y = 0 + pospad
        va = 'bottom'
        ha = 'right'
    elif corner == 2:
        x = 0 + pospad
        y = 0 + pospad
        va = 'bottom'
        ha = 'left'
    elif corner == 3:
        x = 0 + pospad
        y = 1 - pospad
        va = 'top'
        ha = 'left'

    if xy is not None:
        x,y = xy

    fmt = ' ' * lpad + '%s' + ' ' * rpad

    for (t, a) in zip(tag, ax_all):
        a.text(x,
               y,
               fmt % t,
               verticalalignment=va,
               horizontalalignment=ha,
               transform=a.transAxes,
               color=fontcolor,
               bbox={'facecolor': boxcolor,
                     'alpha': alpha,
                     'pad': text_pad,
                     'edgecolor': edgecolor},
               fontsize=fontsize,
               fontweight=fontweight,
               zorder=zorder)

    #--------------------------------------------------
    #marking
    #--------------------------------------------------


def mark_xbounds(ax, bounds, ls=':', color='k', **kwargs):
    """
    mark xbounds with vertical line

    input
    =====
    ax: axes object, or array of axes objects

    bounds: list of bounds

    ls: linestyle (string of list of strings)

    color: (string or list of strings)
    """

    nb = len(bounds)
    ax_all = np.atleast_1d(ax)

    L = list(np.atleast_1d(ls)) * nb

    C = list(np.atleast_1d(color)) * nb

    for b, l, c in zip(bounds, L, C):
        for a in ax_all.flatten():
            a.axvline(x=b, linestyle=l, color=c, **kwargs)

# code to adjust title position
# fig,ax = plt.subplots()
# #ax.set_title('hello')

# ax.title.set_fontsize(12)
# ax.title.set_position((0.5,0.97))
# ax.title.set_verticalalignment('center')
# ax.title.set_backgroundcolor('w')


# #ax.set_title('hello',fontsize=12, position=(0.5,0.97), verticalalignment='center',bbox={'facecolor':'w','edgecolor':'w'})
# #ax.set_title('hello')#,bbox={'facecolor':'w','edgecolor':'w','pad':5})
# #ax.title.set_bbox(facecolor='w',edgecolor='w')#,pad=5)

def set_titles(axs, titles=None,
               x=0.5, y=0.99,
               prefix='',
               va='top', ha='center',
               fontsize=None,
               facecolor='w', edgecolor='w',
               bbox=None,
               zorder=None):


    axs = np.atleast_1d(axs)
    if titles is None:
        titles = [ax.get_title() for ax in axs.flat]
    titles = np.atleast_1d(titles).astype(np.str)

    base_bbox = dict(facecolor=facecolor, edgecolor=edgecolor)
    if bbox is None:
        bbox = {}
    bbox = dict(base_bbox, **bbox)

    for i, (ax, title) in enumerate(zip(axs.flat, titles.flat)):

        if i == 0:
            title = prefix + title

        ax.set_title(title)
        ax.title.set_horizontalalignment(ha)
        ax.title.set_verticalalignment(va)
        ax.title.set_bbox(bbox)
        ax.title.set_position((x, y))

        if fontsize is not None:
            ax.title.set_fontsize(fontsize)

        if zorder is not None:
            ax.title.set_zorder(zorder)


def set_side_titles(axs, titles,
                    x=1.0, y=0.5,
                    prefix='',
                    va='center', ha='center',
                    fontsize='large',
                    bbox=None,
                    facecolor='w', edgecolor='w',
                    rotation=270,
                    **kwargs):

    axs = np.atleast_1d(axs)
    titles = np.atleast_1d(titles).astype(np.str)

    base_bbox = dict(facecolor=facecolor, edgecolor=edgecolor)
    if bbox is None:
        bbox = {}
    bbox = dict(base_bbox, **bbox)

    kwargs = dict(kwargs,
                  va=va, ha=ha,
                  fontsize=fontsize,
                  bbox=bbox,
                  rotation=rotation)

    for i, (ax, title) in enumerate(zip(axs.flat, titles.flat)):
        if i == 0:
            title = prefix + title
        d = dict(dict(transform=ax.transAxes), **kwargs)
        ax.text(x=x, y=y, s=title, **d)


def facetgrid_post(g, adjust_kws=None,
                   top='get',
                   top_kws=None,
                   side='get',
                   side_kws=None):
    """
    adjust facets
    """

    adjust_kws = adjust_kws or {}
    adjust_kws = dict(dict(wspace=0.05, hspace=0.05), **adjust_kws)
    g.fig.subplots_adjust(**adjust_kws)

    if top is not None:
        if len(g.col_names) > 0:
            if top == 'get':
                titles = g.col_names
            else:
                titles = top
            top_kws = top_kws or {}
            set_titles(g.axes[0,:], titles=titles, **top_kws)

    if side is not None:
        if len(g.row_names) > 0:
            if side == 'get':
                titles = g.row_names
            else:
                titles = side
            side_kws = side_kws or {}
            set_side_titles(g.axes[:,-1], titles=titles, **side_kws)



def tag_top(ax,tag,color='k',x=0.5,y=1.0, \
            VA='center',HA='center',\
            fontsize='large',pad=5,facecolor='w',edgecolor='w',alpha=1.0, \
            fontweight='normal', \
            zorder=3,xtrans=None,ytrans=None, **kwargs):
    """
    tag top of axis

    input:
    ax: axis
    tag: string or list
    color: color string or list
    x,y: coordinates

    VA: vertical alignment
    HA: horizontal aligment
    
    xtrans: transform for x coord (data)
    ytrans: transform for y coord (axes)
    """

    if type(tag) is list:
        T = tag
    else:
        T = [tag]

    nt = len(T)

    if type(color) is list:
        C = color * nt
    else:
        C = [color] * nt

    if type(x) is list:
        X = x
    else:
        X = [x]

    if len(T) != len(X):
        print( T, X)
        raise ('len(T)!=len(X)')

    xxtrans = xtrans
    if xxtrans is None:
        xxtrans = ax.transData

    yytrans = ytrans
    if yytrans is None:
        yytrans = ax.transAxes

    trans = _matplotlib.transforms.blended_transform_factory(xxtrans, yytrans)

    for t, c, xx in zip(T, C, X):
        ax.text(xx,y,t,transform=trans, \
                verticalalignment=VA,horizontalalignment=HA, \
                fontsize=fontsize,color=c, fontweight=fontweight,zorder=zorder,\
                bbox={'facecolor':facecolor,'edgecolor':edgecolor,'pad':pad,'alpha':alpha}, **kwargs)







def tag_xbounds(ax, bounds, tag, LB=False, UB=False, **kwds):
    """
    tag bounds with string

    inputs
    ======

    bounds: list of bounds
    tag: string or list of strings

    LB: True if lower bound in bounds (else taken as LB of axis)
    UB: True if upper bound in bounds (else taken as UB of axis)

    **kwds: input to tag_top

    """

    #bounds
    L = list(bounds)
    if not LB:
        L = [ax.axis()[0]] + L

    if not UB:
        L = L + [ax.axis()[1]]

    x = []
    for lb, ub in zip(L[:-1], L[1:]):
        x.append(0.5 * (lb + ub))

    tag_top(ax, tag, x=x, **kwds)


from matplotlib.ticker import MultipleLocator, MaxNLocator

#--------------------------------------------------
#ticks
#--------------------------------------------------


def set_tick_spacing(ax, delta=None, deltaMinor=None, axis=[]):
    """
    set tick spacing to delta

    input
    =====
    ax: ax to alter (or array of axis)

    delta: float spacing (None/False, nothing)
    deltaMinor: float minor spacing (None/False, nothing)
    

    axis: string axis to space ('xaxis','yaxis')
    """

    if delta is None and deltaMinor is None:
        pass
    else:
        ax_all = np.atleast_1d(ax).flatten()
        axis_all = np.atleast_1d(axis)

        for a in ax_all:
            for x in axis_all:
                o = getattr(a, x)

                if delta not in [None, False]:
                    o.set_major_locator(MultipleLocator(delta))
                if deltaMinor not in [None, False]:
                    o.set_minor_locator(MultipleLocator(deltaMinor))


def set_xtick_spacing(ax, delta=None, deltaMinor=None):
    set_tick_spacing(ax, delta, deltaMinor, axis='xaxis')


def set_ytick_spacing(ax, delta=None, deltaMinor=None):
    set_tick_spacing(ax, delta, deltaMinor, axis='yaxis')


def set_tick_nbins(ax, nbins, axis=[]):
    """
    set number of ticks on axis

    input
    =====
    ax: ax or array of ax
    nbins: number of ticks
    axis: string.  axis to alter
    """

    ax_all = np.atleast_1d(ax).flatten()
    axis_all = np.atleast_1d(axis)

    for a in ax_all:
        for x in axis_all:
            o = getattr(a, x)
            o.set_major_locator(MaxNLocator(nbins))


def set_xtick_nbins(ax, nbins):
    set_tick_nbins(ax, nbins, axis='xaxis')


def set_ytick_nbins(ax, nbins):
    set_tick_nbins(ax, nbins, axis='yaxis')


def prune_xticklabels(ax, prune='max', delta=0.1):
    """
    prune tick labels
    

    ax: ax (or arraylike list) 
    prune: type of prune
        'max' (default)
        'min'
        'both'

    delta: fraction of tick section (from upper/lower) to remove

    """
    assert (prune in ['max', 'min', 'both'])

    ax_all = np.atleast_1d(ax).flatten()

    #set default bounds (in data coords)
    LB = []
    UB = []
    for a in ax_all:

        lim = list(a.get_xlim())
        #transform from Axes to Data
        ftrans = lambda x: a.transData.inverted().transform(a.transAxes.transform(x))
        if 'max' in prune or 'both' in prune:
            #prune upper
            lim[1] = ftrans((1. - delta, 0.))[0]

        if 'min' in prune or 'both' in prune:
            #prune lower
            lim[0] = ftrans((0. + delta, 0))[0]
        LB.append(lim[0])
        UB.append(lim[1])

    for (a, lb, ub) in zip(ax_all, LB, UB):

        for pos, label in zip(a.get_xticks(), a.get_xticklabels()):

            if pos < lb:
                label.set_visible(False)
            if pos > ub:
                label.set_visible(False)


def prune_yticklabels(ax, prune='max', delta=0.1):
    """
    prune tick labels
    

    ax: ax (or arraylike list) 
    prune: type of prune
        'max' (default)
        'min'
        'both'

    delta: fraction of tick section (from upper/lower) to remove

    """
    assert (prune in ['max', 'min', 'both'])

    ax_all = np.atleast_1d(ax).flatten()

    #set default bounds (in data coords)
    LB = []
    UB = []
    for a in ax_all:
        lim = list(a.get_ylim())
        #transform from Axes to Data
        ftrans = lambda x: a.transData.inverted().transform(a.transAxes.transform(x))
        if 'max' in prune or 'both' in prune:
            #prune upper
            lim[1] = ftrans((0., 1. - delta))[1]

        if 'min' in prune or 'both' in prune:
            #prune lower
            lim[0] = ftrans((0., 0. + delta))[1]

        LB.append(lim[0])
        UB.append(lim[1])

    for (a, lb, ub) in zip(ax_all, LB, UB):

        for pos, label in zip(a.get_yticks(), a.get_yticklabels()):

            visible = True
            if pos < lb:
                visible = False

            if pos > ub:
                visible = False

            label.set_visible(visible)


def set_minOffset(ax, at='y', delta=0.05):
    """
    adjusts the bounds down by a nice fraction

    ax: single or list of axis objects
    delta: offset for rescaling
    at: which axis to scale
     'x': xaxis
     'y': yaxis
     ['x','y']: both
      
    """

    ax_all = np.atleast_1d(ax).flatten()

    for a in ax_all:

        ftrans = lambda x: a.transData.inverted().transform(a.transAxes.transform(x))

        xmin, ymin = ftrans((0. - delta, 0. - delta))

        if 'x' in at:
            a.set_xlim(left=xmin)
        if 'y' in at:
            a.set_ylim(bottom=ymin)


def set_xminOffset(ax, delta=0.05):
    """
    adjust xlim left by nice fraction
    """
    set_minOffset(ax, 'x', delta)


def set_yminOffset(ax, delta=0.05):
    """
    adjust ylim LB by nice fraciton
    """
    set_minOffset(ax, 'y', delta)


def skip_tick_labels(ax, start=1, step=2, axis=[]):
    """
    skip tick labels

    input:
    ax: axis to alter (or array of ax)

    start: where to start skip
    step : skip every step labels

    axis=('xaxis','yaxis')
    """
    #draw the canvas to get things to work right
    ax.get_figure().canvas.draw()

    ax_all = np.atleast_1d(ax)
    axis_all = np.atleast_1d(axis)

    for a in ax_all:
        for x in axis_all:
            o = getattr(a, x)

            #get first element that is not null
            #oddity in matplotlib that after some rearangement,
            #old ticks are there but null
            for null_start, xx in enumerate(o.get_ticklabels()):
                if xx.get_text() != '':
                    break

            for label in o.get_ticklabels()[null_start + start::step]:
                label.set_visible(False)


def skip_xtick_labels(ax, start=1, step=2):
    skip_tick_labels(ax, start, step, 'xaxis')


def skip_ytick_labels(ax, start=1, step=2):
    skip_tick_labels(ax, start, step, 'yaxis')

#--------------------------------------------------
#labels
#--------------------------------------------------

# def center_ylabel(fig,ax,label,yoffset=0.0,xoffset=-.1,VA='center',HA='center',fontsize='large',**kwd_text):
#     """
#     center label for multiple axes
#     """

#     y=[]
#     for a in ax:
#         y.append(a.get_position().get_points()[:,1])
#     y=np.asarray(y).flatten()

#     ymin,ymax=np.min(y),np.max(y)

#     x=ax[0].get_position().get_points()[0,0]+xoffset

#     yplace=0.5*(ymin+ymax)+yoffset

#     trans = _matplotlib.transforms.blended_transform_factory(ax[0].transAxes,fig.transFigure)

#     t=fig.text(x,yplace,label,transform=trans,verticalalignment='center',horizontalalignment='center', \
#              rotation='vertical',fontsize=fontsize,**kwd_text)
#     return t

# def center_xlabel(fig,ax,label,yoffset=-.1,xoffset=0.0,VA='center',HA='center',fontsize='large',**kwd_text):
#     """
#     center label for multiple axes
#     """

#     x=[]
#     for a in ax:
#         x.append(a.get_position().get_points()[:,0])
#     x=np.asarray(x).flatten()

#     xmin,xmax=np.min(x),np.max(x)

#     y=ax[0].get_position().get_points()[0,1]+yoffset

#     xplace=0.5*(xmin+xmax)+xoffset

#     trans = _matplotlib.transforms.blended_transform_factory(fig.transFigure,ax[0].transAxes)

#     t=fig.text(xplace,y,label,transform=trans,verticalalignment='center',horizontalalignment='center', \
#              fontsize=fontsize,**kwd_text)
#     return t


def center_xlabel(ax,
                  label,
                  yoffset=-.1,
                  xoffset=0.0,
                  fontdict=None,
                  labelpad=None,
                  **kwargs):
    """
    center label for multiple axes
    """
    finv = lambda x: ax[0].transAxes.inverted().transform(x)[0]

    x = []
    for a in ax:
        #x.append(a.get_position().get_points()[:,0])
        #transfrom to axes to figure
        f = lambda x: a.transAxes.transform((x, 0))
        x.append(finv(f(0.)))
        x.append(finv(f(1.)))

    x = np.asarray(x).flatten()

    xmin, xmax = np.min(x), np.max(x)

    y = yoffset

    xplace = 0.5 * (xmin + xmax) + xoffset

    #trans = _matplotlib.transforms.blended_transform_factory(fig.transFigure,ax[0].transAxes)

    # t=ax[0].text(xplace,y,label,transform=ax[0].transAxes,verticalalignment='center',horizontalalignment='center', \
    #          fontsize=fontsize,**kwd_text)
    ax[0].set_xlabel(label, fontdict, labelpad, **kwargs)
    ax[0].xaxis.set_label_coords(xplace, y)


def center_ylabel(ax,
                  label,
                  yoffset=0.0,
                  xoffset=-.1,
                  fontdict=None,
                  labelpad=None,
                  **kwargs):
    """
    center label for multiple axes
    """
    finv = lambda x: ax[0].transAxes.inverted().transform(x)[1]

    y = []
    for a in ax:
        #x.append(a.get_position().get_points()[:,0])
        #transfrom to axes to figure
        f = lambda y: a.transAxes.transform((0, y))
        y.append(finv(f(0.)))
        y.append(finv(f(1.)))

    y = np.asarray(y).flatten()

    ymin, ymax = np.min(y), np.max(y)

    x = xoffset

    yplace = 0.5 * (ymin + ymax) + yoffset

    #trans = _matplotlib.transforms.blended_transform_factory(fig.transFigure,ax[0].transAxes)

    # t=ax[0].text(xplace,y,label,transform=ax[0].transAxes,verticalalignment='center',horizontalalignment='center', \
    #          fontsize=fontsize,**kwd_text)
    ax[0].set_ylabel(label, fontdict, labelpad, **kwargs)
    ax[0].yaxis.set_label_coords(x, yplace)


#super adjuster....
def set_axis_params(ax_in, \
                    xscale=None,xmin=None,xmax=None,xlim=None,dx=None,dxMinor=None, \
                    xskip=None,xprune=None,xtick_pad=None,xlabel=None, \
                    yscale=None,ymin=None,ymax=None,ylim=None,dy=None,dyMinor=None, \
                    yskip=None,yprune=None,ytick_pad=None,ylabel=None, \
                    xoffset=None,yoffset=None,
                    xtickpos=None,ytickpos=None,
                    tick_params=None,
                    lim=None,**kwargs
                    ):
    """
    set axis parameters



    Parameters
    ----------
    ax_in: axis object(s) to adjust

    xscale,yscale: scale (e.g., 'log')

    xtickpos,ytickpos: position of ticks
      x: top,bottom,both,default
      y: left,right,both,default

    tick_params : dict
        if supplied, parameters to ax_in[:].tick_params


    xmin,xmax,ymin,ymax: axis bounds

    dx,dy: tick spacing
    dxMinor,dyMinor: minor tick spacing

    xlim,ylim: 2-tuple
    axis bounds (min,max for x or y)
    Takes precidence over xmin,xmax or ymin,ymax

    lim: 4-tuple of (xmin,xmax,ymin,ymax).  Takes precidents over
    xmin,xmax,ymin,ymax

    xskip,yskip: skip tick labels
      if None/False no skip
      if True: default skip
      if type is int, skip every xskip starting at element 1
      if len==2 with (skip_start,skip_step)

    xtick_pad: padding between axis and ticklabels
      if None/False nothing
      
      
    xprune,yprune: prune labels
      if None/False, no prune
      if True: prune
      if type==str, prune type
      if type==float, prune delta
      if len==2, (prune type,delta)
      prune type x: 'min','max','both'
      prune type y: 'min','max','both'


    xoffset,yoffset: minOffset for lower bound
      if None/False: do nothing
      if True: do default offset
      if type==float: set delta for offset

    """

    if lim is not None:
        assert (len(lim) == 4)
        xmin, xmax, ymin, ymax = lim

    if xlim is not None:
        assert (len(xlim) == 2)
        xmin, xmax = xlim

    if ylim is not None:
        assert (len(ylim) == 2)
        ymin, ymax = ylim

    ax_all = np.atleast_1d(ax_in).flatten()
    for ax in ax_all:

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        if xtickpos is not None:
            ax.xaxis.set_ticks_position(xtickpos)
        if ytickpos is not None:
            ax.yaxis.set_ticks_position(ytickpos)

        if tick_params is not None:
            ax.tick_params(**tick_params)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        set_xtick_spacing(ax, dx, dxMinor)
        set_ytick_spacing(ax, dy, dyMinor)

        #xskip
        if xskip in [None, False]:
            pass
        elif xskip == True:
            skip_xtick_labels(ax)
        elif type(xskip) is int:
            skip_xtick_labels(ax, start=1, step=xskip)
        else:
            assert (len(xskip) == 2)
            skip_xtick_labels(ax, start=xskip[0], step=xskip[1])

        #yskip
        if yskip in [None, False]:
            pass
        elif yskip == True:
            skip_ytick_labels(ax)
        elif type(yskip) is int:
            skip_ytick_labels(ax, start=1, step=yskip)
        else:
            assert (len(yskip) == 2)
            skip_ytick_labels(ax, start=yskip[0], step=yskip[1])

        #pad
        if xtick_pad in [None, False]:
            pass
        else:
            ax.tick_params(axis='x', pad=xtick_pad)

        if ytick_pad in [None, False]:
            pass
        else:
            ax.tick_params(axis='y', pad=ytick_pad)

        #xlabel
        if xlabel in [None, False]:
            pass
        else:
            ax.set_xlabel(xlabel)

        #ylabel
        if ylabel in [None, False]:
            pass
        else:
            ax.set_ylabel(ylabel)

        #xprune
        if xprune in [None, False]:
            pass
        elif xprune == True:
            prune_xticklabels(ax)
        elif type(xprune) is str:
            prune_xticklabels(ax, prune=xprune)
        elif type(xprune) is float:
            prune_xticklabels(ax, delta=xprune)
        else:
            assert (len(xprune) == 2)
            prune_xticklabels(ax, prune=xprune[0], delta=xprune[1])

        #yprune
        if yprune in [None, False]:
            pass
        elif yprune == True:
            prune_yticklabels(ax)
        elif type(yprune) is str:
            prune_yticklabels(ax, prune=yprune)
        elif type(yprune) is float:
            prune_yticklabels(ax, delta=yprune)
        else:
            assert (len(yprune) == 2)
            prune_yticklabels(ax, prune=yprune[0], delta=yprune[1])

            #offset
        if xoffset in [None, False]:
            pass
        elif xoffset == True:
            set_xminOffset(ax)
        elif type(xoffset) is float:
            set_xminOffset(ax, xoffset)

        if yoffset in [None, False]:
            pass
        elif yoffset == True:
            set_yminOffset(ax)
        elif type(yoffset) is float:
            set_yminOffset(ax, yoffset)


from .. import utilities as my_utils


class DefPlot(object):
    """
    generic plot class with defaults
    """

    def __init__(self,
                 clobber=False,
                 axsDic={},
                 pltDic={},
                 legDic={},
                 **kwargs):
        """
        initialize default dictionaries

        *Parameters*

        ax: axis object to associtate with (default None)
        
        axsDic: dictionary of axes parameters
          default dictionary

        axsClobber: bool
          whether to clobber default dic

        pltDic,pltClobber: as above for plot parameters

        legDic,legClobber: as above for legend parameters
        """
        self.axsDic = {}
        self.legDic = {}
        self.pltDic = {}

        self.update_Dics(clobber, axsDic, legDic, pltDic)

    def update_Dics(self, clobber=False, axsDic={}, legDic={}, pltDic={}):
        """
        update all
        """
        self.update_axsDic(clobber, **axsDic)
        self.update_legDic(clobber, **legDic)
        self.update_pltDic(clobber, **pltDic)

    def update_axsDic(self, clobber=False, **kwargs):
        """
        update axsDic
        """
        my_utils.default_dic(
            self.axsDic, kwargs,
            inplace=True, clobber=clobber)

    def update_legDic(self, clobber=False, **kwargs):
        """
        update legDic
        """
        my_utils.default_dic(
            self.legDic, kwargs,
            inplace=True, clobber=clobber)

    def update_pltDic(self, clobber=False, **kwargs):
        """
        update pltDic
        """
        my_utils.default_dic(
            self.pltDic, kwargs,
            inplace=True, clobber=clobber)

    def plotData(self):
        raise AttributeError('to be specified in child')

    def prettyAx(self, ax, clobber=False, **kwargs):
        """
        pretty up axes

        *Parameters*

        ax: axis to pretty up
        
        axsDic: dic
          one off parameters

        axsClobber: bool
          whether to clobber default parameters
        """
        d = kwargs if clobber else dict(self.axsDic, **kwargs)
        set_axis_params(ax, **d)

    def prettyLeg(self, ax, clobber=False, **kwargs):
        """
        pretty up axes

        *Parameters*

        ax: axis to pretty up
        
        legDic: dic
          one off parameters

        legClobber: bool
          whether to clobber default parameters
        """
        d = kwargs if clobber else dict(self.legDic, **kwargs)
        if d:
            ax.legend(**d)

    def __repr__(self):
        return repr(self.__dict__)


from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_standalone_colorbar(ax,
                            cmap,
                            vrange=None,
                            norm=None,
                            cax=None,
                            pos='right',
                            size='3%',
                            pad=0.05,
                            label=None,
                            divider={},
                            colorbar={}):
    """
    create a standalone colorbar attached to axis
    
    Parameters
    ----------
    ax : axis to attach colorbar
    
    cmap : colormap to use
    
    vrange : tuple
        (vmin,vmax).  Used to Normalize colorbar
    
    norm : colormap normalization
        Note, if norm given, ignore vrange
    
    pos : str
        position of colorbar ('right')

    cax : axis or None
     if None, divide axis using size,pad,divider
     if type(cax) is Axes, then use this axis for colorbar
     
    
    size : str
        size of colorbar ('3%')
        
    pad : float
        pad for colorbar (0.05)
   
    divider : dict
        additional kwargs to divider.append_axes
        
    colorbar : dict
        additional kwargs to colorbar
        
    Returns
    -------
    cax : axis for colorbar
    
    cbar : colorbar
    
        
    Example
    -------
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots()
    divider=dict(position='right',size='5%',pad=0.05)
    colorbar=dict(label='hello',ticks=[0.1,0.2,0.3])
    add_standalone_colorbar(ax,plt.cm.cool,norm=plt.Normalize(vmin=0,vmax=1),divider,colorbar)
    
    """

    if label is None:
        d = {}
    else:
        d = dict(label=label)
    colorbar_kwargs = dict(d, **colorbar)

    if norm is None:
        myNorm = plt.Normalize(vmin=vrange[0], vmax=vrange[1])
    else:
        myNorm = norm

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=myNorm)
    sm._A = []

    if cax is None:
        divider_kwargs = dict(
            dict(position=pos, size=size,
                 pad=pad), **divider)
        divider = make_axes_locatable(ax)
        caxUse = divider.append_axes(**divider_kwargs)
    else:
        caxUse = cax

    fig = ax.get_figure()
    cbar = fig.colorbar(sm, cax=caxUse, **colorbar_kwargs)

    return caxUse, cbar

    # def label_line(line, label_text, near_i=None, near_x=None, near_y=None, rotation_offset=0, offset=(0,0)):
    #     """call 
    #         l, = plt.loglog(x, y)
    #         label_line(l, "text", near_x=0.32)
    #     """
    #     def put_label(i):
    #         """put label at given index"""
    #         i = min(i, len(x)-2)
    #         dx = sx[i+1] - sx[i]
    #         dy = sy[i+1] - sy[i]
    #         rotation = np.rad2deg(math.atan2(dy, dx)) + rotation_offset
    #         pos = [(x[i] + x[i+1])/2. + offset[0], (y[i] + y[i+1])/2 + offset[1]]
    #         plt.text(pos[0], pos[1], label_text, size=9, rotation=rotation, color = line.get_color(),
    #         ha="center", va="center", bbox = dict(ec='1',fc='1'))

    #     x = line.get_xdata()
    #     y = line.get_ydata()
    #     ax = line.get_axes()
    #     if ax.get_xscale() == 'log':
    #         sx = np.log10(x)    # screen space
    #     else:
    #         sx = x
    #     if ax.get_yscale() == 'log':
    #         sy = np.log10(y)
    #     else:
    #         sy = y

    #     # find index
    #     if near_i is not None:
    #         i = near_i
    #         if i < 0: # sanitize negative i
    #             i = len(x) + i
    #         put_label(i)
    #     elif near_x is not None:
    #         for i in range(len(x)-2):
    #             if (x[i] < near_x and x[i+1] >= near_x) or (x[i+1] < near_x and x[i] >= near_x):
    #                 put_label(i)
    #     elif near_y is not None:
    #         for i in range(len(y)-2):
    #             if (y[i] < near_y and y[i+1] >= near_y) or (y[i+1] < near_y and y[i] >= near_y):
    #                 put_label(i)
    #     else:
    #         raise ValueError("Need one of near_i, near_x, near_y")


import collections


def label_line(lines,
               labels=None,
               near_frac=None,
               near_i=None,
               near_x=None,
               near_y=None,
               rotation_offset=0,
               offset=(0, 0),
               rotate=True,
               default=None,
               **kwargs):
    """
    add label to line like contour plots

    Parameters
    ----------
    lines: matplotlib.lines.line2D or list of such

    labels : string or None or iterable
         if None, get string from line.get_label()

    near_frac,near_i,near_x,near_y : position of label placement
        frac index,index, x, y

    default : int or None
        if 0, default to first, if -1, default to last, else, drop

    rotation_offset : float
        degrees to offset label

    rotate : bool (Default True)
        if True, do rotation

    offset : tuple (0,0)
        positional offset

    kwargs : extra arguments to ax.text
    """

    bbox = kwargs.pop('bbox',{})
    bbox = dict(dict(ec='1', fc='1'), **bbox)

    kwargs = dict(
        dict(fontsize=9,
             ha="center",
             va="center",
             bbox=bbox), 
        **kwargs)

    def put_label(s, i, x, y, ax, color):
        if i is None:
            return

        if i < 0:
            i = len(x) + i

        i = min(i, len(x) - 2)
        xscreen = ax.transData.transform(list(zip(x[i:i + 2], y[i:i + 2])))
        pos = [(x[i + 1] + x[i]) / 2 + offset[0],
               (y[i + 1] + y[i]) / 2. + offset[1]]
        if rotate:
            rot = np.rad2deg(np.arctan2(*np.abs(np.gradient(xscreen)[0][
                0][::-1]))) + rotation_offset
        else:
            rot = 0.0

        ltex = ax.text(pos[0], pos[1], s, rotation=rot, color=color, **kwargs)
        return ltex

    if not isinstance(lines, collections.Iterable):
        lines = [lines]
    if not isinstance(labels, collections.Iterable):
        if labels is None:
            labels = [None] * len(lines)
        else:
            labels = [labels]

    ret = []
    for line, label in zip(lines, labels):
        ax = line.axes
        color = line.get_color()
        x = line.get_xdata()
        y = line.get_ydata()

        if label is None:
            s = line.get_label()
        else:
            s = label

        # find index
        if near_i is not None:
            ii = near_i
        elif near_frac is not None:
            ii = int(len(x) * near_frac)

        elif near_x is not None:
            ii = default
            for i in range(len(x) - 2):
                if (x[i] < near_x and x[i + 1] >= near_x) or (
                        x[i + 1] < near_x and x[i] >= near_x):
                    ii = i
                    break

        elif near_y is not None:
            ii = default
            for i in range(len(y) - 2):
                if (y[i] < near_y and y[i + 1] >= near_y) or (
                        y[i + 1] < near_y and y[i] >= near_y):
                    ii = i
                    break

        else:
            raise ValueError("Need one of near_i, near_frac, near_x, near_y")

        #ret.append(put_label(s,i,x,y,ax,color))
        put_label(s, ii, x, y, ax, color)



def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = _matplotlib.colors.LinearSegmentedColormap(name, cdict)
    #plt.register_cmap(cmap=newcmap)

    return newcmap
