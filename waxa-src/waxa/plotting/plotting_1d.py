import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np
import numpy.typing as npt
from waxa.helper import xlabels_1d
from waxa.helper.datasmith import key_from_attribute, sort
# Unit detection lives in waxa.plotting.units; re-exported here because many
# notebooks and kexp import it from this module.
from .units import (UNIT_MAP_FROM_COMMENT, _normalize_name, get_param,
                    guess_unit, detect_unit)

__all__ = [
    'errorplot',
    'get_param',
    'guess_unit',
    'detect_unit',
    'plot_mixOD',
    'plot_sum_od_fits',
    'plot_fit_residuals',
    'sort',
]

def errorplot(ad, mean=None, yerr=None, y=None,
              ymult = 1., yunit = None):

    unit, mult, xvarname = detect_unit(ad, xvar_idx=0)

    fig, axs = plt.subplots(1,1, figsize=(4, 3), layout='constrained')
    axs = np.atleast_1d(axs)

    if mean is not None:
        if yerr is None:
            yerr = np.zeros_like(mean)
        axs[0].errorbar(
            ad.avg.xvars[0] * mult,
            mean * ymult,
            yerr=yerr * ymult,
            fmt='o-',lw=1,ms=4)

    if y is not None:
        axs[0].scatter(ad.xvars[0] * mult,
                    y * ymult,
                    s=10, zorder=5, alpha=0.25)

    key = key_from_attribute(ad, y)
    axs[0].set_xlabel(f"{xvarname} ({unit})")
    ylabel = f"{key} ({yunit})" if yunit is not None else f"{key}"
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(f"run {ad.run_info.run_id}")

    return fig, axs

def plot_mixOD(ad,
               ndarray=[],
               xvar_idx=0,
               xvarformat="1.2f",
               xvarmult = 1.,
               xvarunit = "",
               lines=False,
               max_od=0.,
               figsize=[],
               aspect='auto',
               swap_axes=None):
    # Extract necessary information

    from waxa import atomdata
    ad: atomdata
    
    xvarnames = ad.xvarnames
    xvars = ad.xvars
    xvarunit, xvarmult, xvarname = detect_unit(ad, xvar_idx, xvarunit=xvarunit, xvarmult=xvarmult) ##

    if isinstance(ndarray,np.ndarray):
        od = ndarray
    else:
        od = ad.od

    if max_od == 0.:
        max_od = np.max(od)

    # Calculate the dimensions of the stitched image
    n, px, py = od.shape
    if isinstance(ad.params.N_repeats,np.ndarray):
        if ad.params.N_repeats.size > 1:
            n_repeats = 1
        else:
            n_repeats = int(ad.params.N_repeats)
    else:
        n_repeats = int(ad.params.N_repeats)
    n_shots = int(n / n_repeats)

    # Auto-detect swap_axes if not explicitly set
    if swap_axes is None:
        if n_shots == 1 and n_repeats > 1:
            swap_axes = True
        else:
            swap_axes = False

    if swap_axes:
        total_width = n_repeats * px
        max_height = n_shots * py
    else:
        total_width = n_shots * px
        max_height = n_repeats * py
        
    # Create a figure and axis for plotting
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    # Initialize x position for each image
    x_pos = 0
    y_pos = 0

    # print(n_shots)
    # print(n_repeats)

    # Plot each image and label with xvar value
    if swap_axes:
        for i in range(n_repeats):
            for j in range(n_shots):
                idx = i + j*n_repeats
                img = od[idx]
                ax.imshow(img, extent=[x_pos, x_pos+px, y_pos, y_pos+py],
                        vmin=0.,vmax=max_od, origin='lower')
                ax.axvline()
                y_pos += py
            y_pos = 0
            x_pos += px
    else:
        for i in range(n_shots):
            for j in range(n_repeats):
                idx = j + i*n_repeats
                img = od[idx]
                ax.imshow(img, extent=[x_pos, x_pos+px, y_pos, y_pos+py],
                        vmin=0.,vmax=max_od, origin='lower')
                ax.axvline()
                y_pos += py
            y_pos = 0
            x_pos += px

    # Add lines between images if requested
    if lines:
        if swap_axes:
            # Draw horizontal lines between rows
            for pos in np.arange(py, max_height, py):
                ax.axhline(pos, color='white', linewidth=1)
            # Draw vertical lines between columns
            for pos in np.arange(px, total_width, px):
                ax.axvline(pos, color='white', linewidth=1)
        else:
            # Draw vertical lines between columns
            for pos in np.arange(px, total_width, px):
                ax.axvline(pos, color='white', linewidth=1)
            # Draw horizontal lines between rows
            for pos in np.arange(py, max_height, py):
                ax.axhline(pos, color='white', linewidth=1)

    plt.gca().set_aspect(aspect)

    # Set axis labels and title
    label_name = xvarname
    axislabel_str = f'{label_name}'
    if xvarunit != "":
        axislabel_str += f' ({xvarunit})'    
    ax.set_title(f"Run ID: {ad.run_info.run_id}")

    # Set the x-axis limits to show all images
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, max_height)

    if swap_axes:
        # Remove x-axis ticks and labels
        ax.xaxis.set_visible(False)
        ax.yaxis.set_ticks([])
    else:
        # Remove y-axis ticks and labels
        ax.yaxis.set_visible(False)
        ax.xaxis.set_ticks([])

    axislabel_str = f'{xvarnames[xvar_idx]}'
    if xvarunit != "":
        axislabel_str += f' ({xvarunit})'

    if swap_axes:
        ax.set_ylabel(axislabel_str)
        # Set ticks at the center of each sub-image and rotate them vertically
        tick_positions = np.arange(py/2, max_height, py)
        ax.set_yticks(tick_positions)
        xvarlabels = xlabels_1d(xvars[xvar_idx], xvarmult, xvarformat)
        xvarlabels = xvarlabels[::n_repeats]
        ax.set_yticklabels(xvarlabels, rotation='vertical', va='center')
        plt.minorticks_off()
    else:
        ax.set_xlabel(axislabel_str)
        # Set ticks at the center of each sub-image and rotate them vertically
        tick_positions = np.arange(px/2, total_width, px)
        ax.set_xticks(tick_positions)
        xvarlabels = xlabels_1d(xvars[xvar_idx], xvarmult, xvarformat)
        xvarlabels = xvarlabels[::n_repeats]
        ax.set_xticklabels(xvarlabels, rotation='vertical', ha='center')
        plt.minorticks_off()

    if lines:
        for pos in np.arange(px, total_width, px):
            ax.axvline(pos, color='white', linewidth=1)

    # Show the plot
    fig.tight_layout()

def plot_sum_od_fits(ad,axis=0,
                    xvarformat=None,
                    xvarmult=None,
                    figsize=[],
                    **kwargs):
    # A 2D scan holds a grid of fits (one per xvar0/xvar1 pair), so the flat
    # indexing below would iterate over single fit objects.  Hand those off.
    if getattr(ad, 'Nvars', len(ad.xvars)) > 1:
        from .plotting_2d import plot_sum_od_fits_grid
        if xvarformat is not None:
            kwargs['xvarformat'] = xvarformat
        if xvarmult is not None:
            kwargs.setdefault('xvar0mult', xvarmult)
            kwargs.setdefault('xvar1mult', xvarmult)
        return plot_sum_od_fits_grid(ad, axis=axis, figsize=figsize, **kwargs)

    if kwargs:
        raise TypeError(f"plot_sum_od_fits() got unexpected keyword arguments "
                        f"{sorted(kwargs)} for a 1D scan")
    if xvarformat is None:
        xvarformat = '3.3g'
    if xvarmult is None:
        xvarmult = 1.

    if axis == 0:
        fits = ad.cloudfit_x
        label = "x"
    elif axis == 1:
        fits = ad.cloudfit_y
        label = "y"
    else:
        raise ValueError("Axis must be 0 (x) or 1 (y)")
    
    ymax = np.max([np.max(fit.ydata) for fit in fits])

    if isinstance(ad.params.N_repeats,np.ndarray):
        ad.params.N_repeats = ad.params.N_repeats[0]

    Nr = ad.params.N_repeats
    Ns = int(len(ad.xvars[0]) / Nr)

    if figsize:
        fig, ax = plt.subplots(Nr,Ns,
                               figsize=figsize,
                               layout='constrained')
    else:
        fig, ax = plt.subplots(Nr,Ns,
                           layout='constrained')

    if isinstance(ax, Axes):
        ax = [ax]

    xvar = ad.xvars[0]
    xvarlabels = xlabels_1d(xvar, xvarmult, xvarformat)

    if Nr == 1 or Ns == 1:
        for i in range(max(Nr,Ns)):

            yfit = fits[i].y_fitdata
            ydata = fits[i].ydata
            xdata = fits[i].xdata

            ax[i].plot(xdata*1.e6,ydata)
            ax[i].plot(xdata*1.e6,yfit)
            ax[i].set_ylim([0,1.1*ymax])

            ax[i].set_xlabel(xvarlabels[i],rotation='vertical')

            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else:
        for i in range(Ns):
            for j in range(Nr):
                idx = j + i*Nr

                yfit = fits[idx].y_fitdata
                ydata = fits[idx].ydata
                xdata = fits[idx].xdata

                ax[j,i].plot(xdata*1.e6,ydata)
                ax[j,i].plot(xdata*1.e6,yfit)
                ax[j,i].set_ylim([0,1.1*ymax])

                ax[j,i].set_xticks([])
                ax[j,i].set_yticks([])

                if j == Nr-1:
                    ax[j,i].set_xlabel(xvarlabels[idx],rotation='vertical')
                    
    fig.suptitle(f"Run ID: {ad.run_info.run_id}\nsum_od_{label}")
    fig.supxlabel(ad.xvarnames[0])

def plot_fit_residuals(ad,axis=0,
                       xvarformat='1.3g',
                        xvarmult=1.,
                        figsize=[]):
    if axis == 0:
        fits = ad.cloudfit_x
        label = "x"
    elif axis == 1:
        fits = ad.cloudfit_y
        label = "y"
    else:
        raise ValueError("Axis must be 0 (x) or 1 (y)")
    
    if isinstance(ad.params.N_repeats,np.ndarray):
        ad.params.N_repeats = ad.params.N_repeats[0]

    fits_yfitdata = [fit.y_fitdata for fit in fits]
    fits_ydata = [fit.ydata for fit in fits]
    xdata = fits[0].xdata
    sum_od_residuals = np.asarray(fits_ydata) - np.asarray(fits_yfitdata)
    print(sum_od_residuals.shape)

    if figsize:
        fig, ax = plt.subplots(ad.params.N_repeats,ad.params.N_shots,
                               figsize=figsize)
    else:
        fig, ax = plt.subplots(ad.params.N_repeats,ad.params.N_shots)

    bools = ~np.isinf(sum_od_residuals) & ~np.isnan(sum_od_residuals)
    ylimmin = np.min(sum_od_residuals[bools])
    ylimmax = np.max(sum_od_residuals[bools])

    Nr = ad.params.N_repeats
    Ns = ad.params.N_shots

    xvar = ad.xvars[0]
    xvarlabels = xlabels_1d(xvar, xvarmult, xvarformat)

    if ad.params.N_repeats == 1:
        for i in range(Ns):
            ax[i].plot(xdata,sum_od_residuals[i])

            ax[i].set_xlabel(xvarlabels[i],rotation='vertical')
  
            ax[i].set_ylim(ylimmin,ylimmax)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
    else:
        for j in range(Nr):
            for i in range(Ns):
                idx = j + i*Nr
                ax[j,i].plot(xdata,sum_od_residuals[idx])
                ax[j,i].set_xlabel(xvarlabels[idx])
                ax[j,i].set_ylim(ylimmin,ylimmax)

                ax[j,i].set_xticks([])
                
                if i != 0:
                    ax[j,i].set_yticklabels([])
                else:
                    ax[j,i].set_yticks([])

                if j == Nr-1:
                    ax[j,i].set_xlabel(xvarlabels[idx],rotation='vertical')

    fig.suptitle(f"Run ID: {ad.run_info.run_id}\nsum_od_{label} fit residuals")
    fig.supxlabel(ad.xvarnames[0])
    fig.set_figwidth(18)
    fig.tight_layout()

    plt.show()

