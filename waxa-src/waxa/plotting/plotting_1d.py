import matplotlib.pyplot as plt
import numpy as np
from waxa import atomdata
from waxa.helper import xlabels_1d
import inspect
import re
def _normalize_name(name):
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return name.strip().strip("\x00")


UNIT_MAP_FROM_COMMENT = {
    "ns":        ("ns", 1e9),
    "us":        ("µs", 1e6),
    "µs":        ("µs", 1e6),
    "ms":        ("ms", 1e3),
    "s":         ("s", 1.0),
    "MHz":       ("MHz", 1e-6),
    "kHz":       ("kHz", 1e-3),
    "Hz":        ("Hz", 1.0),
    "Gamma":     ("Γ", 1.0),
    "V":         ("V", 1.0),
    "A":         ("A", 1.0),
    "amplitude": ("", 1.0),
    "fraction":  ("", 1.0),
    "rad":       ("π", 1 / np.pi),
    "unitless":  ("", 1.0),
}


def get_param(params_obj, param_name):
    param_name = _normalize_name(param_name)

    try:
        src = inspect.getsource(params_obj.__class__)
    except OSError:
        return None, 1.0
    pattern = rf"self\.{re.escape(param_name)}\s*=\s*.*?#\s*([^\n]+)"
    m = re.search(pattern, src)
    if not m:
        return None, 1.0

    raw_unit = m.group(1).strip()

    for key, (unit_label, mult) in UNIT_MAP_FROM_COMMENT.items():
        if key in raw_unit:
            return unit_label, mult

    return None, 1.0


def guess_unit(name, values):

    name = _normalize_name(name)
    lname = name.lower()

    try:
        vals = np.asarray(values, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None, 1.0
        vmax = float(np.max(np.abs(vals)))
    except Exception:
        return None, 1.0

    # Time 
    if lname.startswith("t_") or "time" in lname or lname.endswith("_t"):
        if vmax >= 1:
            return "s", 1.0
        elif vmax >= 1e-3:
            return "ms", 1e3
        elif vmax >= 1e-6:
            return "µs", 1e6
        elif vmax >= 1e-9:
            return "ns", 1e9
        else:
            return "s", 1.0

    # Frequency
    if ("freq" in lname or "frequency" in lname or "f_" in lname or
            "raman" in lname or "rf" in lname):
        if vmax >= 1e9:
            return "GHz", 1e-9
        elif vmax >= 1e6:
            return "MHz", 1e-6
        elif vmax >= 1e3:
            return "kHz", 1e-3
        else:
            return "Hz", 1.0

    # detuning in units of Gamma Γ
    if lname.startswith("detune_") or "detun_" in lname:
        return "Γ", 1.0

    # Voltage
    if lname.startswith("v_") or "volt" in lname:
        return "V", 1.0

    # Current
    if lname.startswith("i_") or "current" in lname:
        return "A", 1.0

    # Amplitude / power fraction (dimensionless)
    if (lname.startswith("amp_") or
            lname.startswith("pfrac_") or "fraction" in lname):
        return "(amp)", 1.0
    
    if (lname.startswith("phase_")):
        return "π", 1/np.pi

    # Default: unknown / unitless
    return None, 1.0


def detect_unit(ad: atomdata, xvar_idx, xvarunit="", xvarmult=1.0):

    xvarname = _normalize_name(ad.xvarnames[xvar_idx])
    xvar_vals = ad.xvars[xvar_idx]

    unit_from_comment, mult_from_comment = get_param(ad.params, xvarname)

    source = "comment"
    if unit_from_comment is None:
        unit_from_comment, mult_from_comment = guess_unit(xvarname, xvar_vals)
        source = "guess"

    final_unit = xvarunit if xvarunit != "" else (unit_from_comment or "")

    final_mult = xvarmult if xvarmult != 1.0 else (mult_from_comment or 1.0)

    print(f"xvar = {xvarname}, source = {source}, "
          f"detected unit = {unit_from_comment}, multiplier = {mult_from_comment:.1e}")
    print(f"final xvarunit = {final_unit}, xvarmult = {final_mult:.1e}")

    return final_unit, final_mult, xvarname

def plot_mixOD(ad:atomdata,
               ndarray=[],
               xvar_idx=0,
               xvarformat="1.2f",
               xvarmult = 1.,
               xvarunit = "",
               lines=False,
               max_od=0.,
               figsize=[],
               aspect='auto',
               swap_axes=False):
    # Extract necessary information
    
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

def plot_sum_od_fits(ad:atomdata,axis=0,
                    xvarformat='3.3g',
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
    
    ymax = np.max([np.max(fit.ydata) for fit in fits])

    if isinstance(ad.params.N_repeats,np.ndarray):
        ad.params.N_repeats = ad.params.N_repeats[0]

    Nr = ad.params.N_repeats
    Ns = int(len(ad.xvars[0]) / Nr)

    if figsize:
        fig, ax = plt.subplots(Nr,Ns,
                               figsize=figsize,layout='tight')
    else:
        fig, ax = plt.subplots(Nr,Ns,
                           layout='tight')

    

    xvar = ad.xvars[0]
    xvarlabels = xlabels_1d(xvar, xvarmult, xvarformat)

    if ad.params.N_repeats == 1 or Ns == 1:
        for i in range(Ns):

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

def plot_fit_residuals(ad:atomdata,axis=0,
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

