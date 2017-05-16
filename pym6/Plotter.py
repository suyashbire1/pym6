import numpy as np
import matplotlib.pyplot as plt
import pyximport
pyximport.install()
from .getvaratz import getvaratz as _vatz, getTatz as _Tatz, getTatzlin as _Tatzlin 

def rhotoz(var,z,e,**kwargs):
    input_dtype = var.dtype
    z = np.array(z,dtype=np.float32)
    if z.ndim == 0:
        z = z[np.newaxis]
    assert var.ndim == 4 or var.ndim == 1
    if var.ndim == 4:
        assert var.shape[1] == e.shape[1]-1
        assert var.loc[0] == e.loc[0]
        var = var.astype(np.float32)
        e = e.astype(np.float32)
        output_var = _vatz(var,z,e)
        output_var = output_var.astype(np.float32)
    elif var.ndim == 1:
        kwargs.get('lin_interp',False)
        var = var.astype(np.float32)
        e = e.astype(np.float32)
        if lin_interp:
            output_var = _Tatzlin(var,z,e)
        else:
            output_var = _Tatz(var,z,e)
    return output_var.astype(input_dtype)

def plotter(self,reduce_func,mean_axes,plot_kwargs={},**kwargs):

    gridloc = ['u','v','h','q']  # Four vertices of an Arakawa C-grid cell
    xloc = ['q','h','h','q']     # X Location on the grid for variable in gridloc
    yloc = ['h','q','h','q']     # Y Location on the grid for variable in gridloc

    xloc_yloc_index = gridloc.index(self.values.loc[0])
    X = getattr(self.dom,'lon'+xloc[xloc_yloc_index])
    Y = getattr(self.dom,'lat'+xloc[xloc_yloc_index])

    gridloc = ['l','i']
    zloc = ['Layer','Interface']
    zloc_index = gridloc.index(self.values.loc[1])
    Z = getattr(self.dom,zloc[zloc_index])
    T = self.Time
    axes_label = ['Time',r'$\rho$','Lat','Lon']
    axes_units = ['s',r'$kgm^{-3}$',r'$^{\circ}$',r'$^{\circ}$']
    axes = [T,Z,Y,X]
    for i,axis in enumerate(axes):
        axes[i] = axis[self._plot_slice[i,0]:self._plot_slice[i,1]]

    keep_axes = ()
    for i in range(4):
        if i not in mean_axes:
            keep_axes += (i,)

    values = self.values
    if 1 in keep_axes:
        zcoord = kwargs.get('zcoord',False)
        if zcoord:
            z = kwargs.get('z')
            e = kwargs.get('e')
            axes_label[1] = 'z'
            axes_units[1] = 'm'
            axes[1] = z
            isop_mean = kwargs.get('isop_mean',True)
            e = e.values
            if isop_mean:
                values = getattr(np,reduce_func)(values,
                        axis=mean_axes,keepdims=True)
                e = getattr(np,reduce_func)(e,
                        axis=mean_axes,keepdims=True)
            values = rhotoz(values,z,e)
    elif 2 in keep_axes and 3 in keep_axes:
        hslice = kwargs.get('hslice',False)
        if hslice:
            z = kwargs.get('z')
            e = kwargs.get('e')
            e = e.values
            values = rhotoz(values,z,e)

    values = getattr(np,reduce_func)(values,axis = mean_axes)

    ax = kwargs.get('ax',None)
    if ax is None:
        fig,ax = plt.subplots(1,1)

    if len(keep_axes) == 1:
        i = keep_axes[0]
        x = axes[i]
        im = ax.plot(x,values,**plot_kwargs)
        ax.set_xlabel(axes_label[i] + ' (' + axes_units[i] + ')')
        if self.name and self.units:
            ax.set_ylabel(self.name + ' (' + self.units + ')')

    elif len(keep_axes) == 2:
        i = keep_axes[0]
        ylabel = axes_label[i] + ' (' + axes_units[i] + ')'
        y = axes[i]
        i = keep_axes[1]
        xlabel = axes_label[i] + ' (' + axes_units[i] + ')'
        x = axes[i]

        perc = kwargs.get('perc',None)
        if perc:
            vlim = np.nanpercentile(np.fabs(values),perc)
            vmin,vmax = -vlim,vlim
            if 'vmin' in plot_kwargs:
                vmin = plot_kwargs.get('vmin')
                plot_kwargs.pop('vmin')
            if 'vmax' in plot_kwargs:
                vmax = plot_kwargs.get('vmax')
                plot_kwargs.pop('vmax')

        dx = np.diff(x)[0]
        dy = np.diff(y)[0]
        extent = [x.min()-dx/2,x.max()+dx/2,y.min()-dy/2,y.max()+dy/2]
        im = ax.imshow(values,origin='lower',extent=extent,
                       vmin=vmin,vmax=vmax,
                       interpolation='none', aspect='auto',
                       **plot_kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        cbar = kwargs.get('cbar',False)
        if cbar:
            cbar = plt.colorbar(im,ax=ax)
            cbar.formatter.set_powerlimits((-3, 4))
            cbar.update_ticks()
        ax.set_xlim(x.min(),x.max())
        ax.set_ylim(y.min(),y.max())
    return im