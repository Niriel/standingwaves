"""
.. module:: load_data
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Bertrand Delforge <b.delforge@sron.nl>

"""
USE_PS = True
import numpy as np
if USE_PS:
    import matplotlib
    matplotlib.use('ps')
import matplotlib.pyplot as plt
import my_style

def smooth(x, window_len=11, window='hanning'):
        if x.ndim != 1:
                raise ValueError, "smooth only accepts 1 dimension arrays."
        if x.size < window_len:
                raise ValueError, "Input vector needs to be bigger than window size."
        if window_len < 3:
                return x
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
                raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
        s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
        if window == 'flat':  # moving average
                w = np.ones(window_len, 'd')
        else:
                w = eval('np.' + window + '(window_len)')
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len:-window_len + 1]

filenames = sorted("""
1028.9385.csv  1029.6225.csv  1030.4055.csv  1031.2155.csv  1031.9535.csv  1032.7410.csv  1041.3855.csv  1042.0875.csv  1042.8975.csv  1043.6985.csv  1044.5040.csv
1029.0195.csv  1029.8115.csv  1030.5900.csv  1031.3640.csv  1032.1830.csv  1032.9435.csv  1041.5655.csv  1042.2855.csv  1043.0460.csv  1043.8605.csv  1044.6750.csv
1029.1500.csv  1029.9600.csv  1030.7790.csv  1031.5395.csv  1032.2550.csv  1040.8815.csv  1041.6420.csv  1042.4610.csv  1043.2350.csv  1044.0135.csv  1044.8055.csv
1029.3210.csv  1030.1265.csv  1030.9275.csv  1031.7375.csv  1032.4395.csv  1041.0840.csv  1041.8715.csv  1042.6095.csv  1043.4195.csv  1044.2025.csv  1044.8865.csv
1029.4965.csv  1030.3155.csv  1031.0805.csv  1031.8545.csv  1032.6105.csv  1041.2145.csv  1041.9705.csv  1042.7445.csv  1043.5050.csv  1044.3240.csv
""".split())

def load(filename):
    lo = float(filename[:-4])
    with open(filename) as f:
        lines = f.readlines()
    line_x, line_y = lines
    xss = line_x.split()
    yss = line_y.split()
    xs = map(float, xss)
    ys = map(float, yss)
    x = np.array(xs)
    y = np.array(ys)
    order = x.argsort()
    x = x[order]
    y = y[order]
    return lo, x, y


with my_style.latex(my_style.WIDTH_ARTICLE):
    my_style.pretty(1)
    plt.subplot(1, 2, 1)
    los = []
    maxes = []
    noises = []
    color_id = 0
    for filename in filenames:
        lo, x, y = load(filename)
        to_keep = x < 1042
        x = x[to_keep]
        y = y[to_keep]
        if len(x) <= 1:
            continue
        baseline = np.median(y)
        continuum_idx = y - baseline < .5
        los.append(lo)
        maxes.append(np.max(smooth(y - baseline)))
        noises.append(np.std(y[continuum_idx]))
        plt.plot(x, y)
    xmin = 1037.0
    xmax = 1037.2
    ticks_nb = 5
    ticks_pos = np.linspace(xmin, xmax, ticks_nb)
    ticks_lbl = ['%07.2f' % tick_pos for tick_pos in ticks_pos]
    ticks_lbl = map(my_style.latexM, ticks_lbl)
    plt.xlim(xmin, xmax)
    plt.xticks(ticks_pos, ticks_lbl)
    plt.xlabel(my_style.latexT("USB Frequency [GHz]"))
    plt.ylabel(my_style.latexT("Flux [K]"))
    plt.title(my_style.latexT("Line profile"))

    #
    plt.subplot(1, 2, 2)
    los = np.array(los[2:])
    print len(los)
    maxes = np.array(maxes[2:])
    noises = np.array(noises[2:])
    plt.errorbar(los, maxes, yerr=noises, fmt='o', color=my_style.COLORS_STD[0])
    plt.xlabel(my_style.latexT("LO Frequency [GHz]"))
    plt.ylabel(my_style.latexT("Flux [K]"))
    xmin = np.min(los)
    xmax = np.max(los)
    ticks_nb = 5
    ticks_pos = np.linspace(xmin, xmax, ticks_nb)
    ticks_lbl = ['%04.0f' % tick_pos for tick_pos in ticks_pos]
    ticks_lbl = map(my_style.latexM, ticks_lbl)
    plt.xlim(xmin, xmax)
    plt.xticks(ticks_pos, ticks_lbl)
    plt.xlabel(my_style.latexT("LO frequency [GHz]"))
    plt.ylabel("")
    plt.title(my_style.latexT("Line peak intensity"))
    #
    scatter = np.std(maxes)
    noise = np.median(noises)
    print scatter / noise

    if USE_PS:
        plt.savefig('obsid_5000352C.eps', bbox_inches='tight')
    else:
        plt.show()
