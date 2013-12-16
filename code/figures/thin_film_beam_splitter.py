from __future__ import division
USE_PDF = True
import numpy as np
import matplotlib
if USE_PDF:
    matplotlib.use('pdf')
import matplotlib.pyplot as plt
import scipy.constants

import style.my_style as style
import sw

Z0 = scipy.constants.value('characteristic impedance of vacuum')


#                               Sky
#                                |
#         LO                     6
# [source]0 --- 1[space]2 --- 3[film]5
#                                4
#                                |
#                                7
#                              [space]
#                                8
#                                |
#                                9
#                            [receiver]
#                               10
lo_freq = 500e9

def compute():
    couplings = sw.ExpandCouplingsTo3d(({0, 1}, {2, 3}, {4, 7}, {8, 9}))
    n_ports = 11 * 3
    presolve = sw.Solver(n_ports, couplings)

    # Frequency independent networks.
    # -------------------------------

    # Let's say that the source reflects 10 % of the incoming power.
    # x^2 = .1, x = sqrt(.1).
    # Also the source has a higher index, so we put a minus sign.
    source_r = -np.sqrt(.1)
    source = sw.networks.MirrorNormal(source_r)

    # Receiver also reflects 10 %.
    receiver_r = -np.sqrt(.1)
    receiver_t = np.sqrt(.9)
    receiver = sw.networks.SemiTransparentMirrorNormal(receiver_r, receiver_t)

    networks = [source, None, None, None, receiver]

    n1 = 1.00
    n2 = 1.83 + 0.018j  # Turn off imaginary part when making thick films.
    l2 = 10e-6  # Thickness of the film.
    tb = sw.geometry.TaitBryan(0, 1, 2)
    k1 = np.array([0, 0, 1])
    att = (-sw.geometry.TAU / 8, 0, 0)

    lo_power = 100
    sky_power = 1
    lo_field = (2 * Z0 * lo_power) ** .5
    sky_field = (2 * Z0 * sky_power) ** .5
    a_lo = np.zeros(n_ports, dtype=complex)
    a_sky = np.zeros(n_ports, dtype=complex)
    c_lo = np.zeros(n_ports, dtype=complex)
    c_sky = np.zeros(n_ports, dtype=complex)
    c_lo[0:0 + 3] = np.array([1, 1, 0]) * lo_field
    a_sky[6 * 3:6 * 3 + 3] = np.array([1, 0, 1]) * sky_field

    # Frequency loop.
    n_frequencies = 1001
    if_min = 4e9
    if_max = if_min + 4e9
    frequencies_lsb = np.linspace(lo_freq - if_max, lo_freq - if_min, n_frequencies)[::-1]
    frequencies_usb = np.linspace(lo_freq + if_min, lo_freq + if_max, n_frequencies)
    frequencies = np.concatenate((frequencies_lsb, frequencies_usb))
    b_lo = np.zeros((len(frequencies), n_ports), dtype=complex)
    b_sky = np.zeros((len(frequencies), n_ports), dtype=complex)
    for i, frequency in enumerate(frequencies):
        networks[1] = sw.networks.Distance(n1, .7, frequency)
        networks[2], _, _, _ = sw.networks.ThinFilmOblique(n1, n2, l2, frequency, tb, att, k1)
        networks[3] = sw.networks.Distance(n1, .3, frequency)
        solve = presolve(networks)
        b_lo[i, :] = solve(a_lo, c_lo)
        b_sky[i, :] = solve(a_sky, c_sky)
    return frequencies, b_lo, b_sky

def plot(width, frequencies, b_lo, b_sky):
    # Plot unfolded.
    lsb = frequencies < lo_freq
    usb = frequencies > lo_freq
    xs = frequencies / 1e9

    def power(field):
        return np.abs(field) ** 2 / Z0 / 2

    lo_h_f = b_lo[:, 10 * 3 + 0]
    lo_v_f = b_lo[:, 10 * 3 + 2]
    sky_h_f = b_sky[:, 10 * 3 + 0]
    sky_v_f = b_sky[:, 10 * 3 + 2]

    lo_h = power(lo_h_f)
    lo_v = power(lo_v_f)
    sky_h = power(sky_h_f)
    sky_v = power(sky_v_f)

    with style.latex(width, scipy.constants.golden):
        style.pretty()
        plt.figure(0)
        plt.axvspan(np.max(xs[lsb]),
                    np.min(xs[usb]),
                    facecolor='#aaaaaa',
                    alpha=0.5)
        plt.plot(xs[lsb], lo_h[lsb], '-', label=style.latexT("LO H"), color=style.COLORS_STD[1])
        plt.plot(xs[lsb], lo_v[lsb], '--', label=style.latexT("LO V"), color=style.COLORS_STD[1])
        plt.plot(xs[lsb], sky_h[lsb], '-', label=style.latexT("Sky H"), color=style.COLORS_STD[2])
        plt.plot(xs[lsb], sky_v[lsb], '--', label=style.latexT("Sky V"), color=style.COLORS_STD[2])
        plt.plot(xs[usb], lo_h[usb], '-', color=style.COLORS_STD[1])
        plt.plot(xs[usb], lo_v[usb], '--', color=style.COLORS_STD[1])
        plt.plot(xs[usb], sky_h[usb], '-', color=style.COLORS_STD[2])
        plt.plot(xs[usb], sky_v[usb], '--', color=style.COLORS_STD[2])
        plt.annotate(style.latexT("LSB"),
                     xy=(np.mean(xs[lsb]), 2),
                     horizontalalignment='center', verticalalignment='center')
        plt.annotate(style.latexT("USB"),
                     xy=(np.mean(xs[usb]), 2),
                     horizontalalignment='center', verticalalignment='center')
        plt.legend(loc='center', prop={"size":6})
        plt.xlabel(style.latexT("Frequency [\\si{\giga\hertz}]"))
        plt.ylabel(style.latexT("Flux density  [\\si{\watt\per\meter\squared}]"))
        if USE_PDF:
            plt.savefig("thin_film_beam_splitter_detailed.pdf", bbox_inches='tight')
        else:
            plt.show()

    # Compute folded.
    x_folded = xs[usb] - xs[usb][0]
    y_folded_h = lo_h[lsb] + sky_h[lsb] + lo_h[usb] + sky_h[usb]
    y_folded_v = lo_v[lsb] + sky_v[lsb] + lo_v[usb] + sky_v[usb]

    # Plot folded.

    # With my previous model, I used to see a beat.  The folded signal was not
    # at the same period as the original signal.  I was saying that this was
    # because there were actually two lo-mixer cavities: one using the near
    # side, and one using the far side of the beam splitter.  Each cavity had
    # its own period, and once I was folding it would somehow show drastically.
    # This was, I believe, an error on my part.  Indeed, I was adding the LSB
    # and USB fields, then raising it to a power.  I believe now that this is
    # wrong: there is no reason for the LSB and USB fields to be phase-locked,
    # therefore they do not add coherently. I must raise them to power before
    # adding them.  When doing that, there is no obvious beat.  HOWEVER, when
    # using pathological dimensions for the 'thin film' (like, 20 cm thick) and
    # a broad enough frequency range, you can see this beat.  So my model DOES
    # take into account both sides of the thin film, it's just that it does not
    # show much.  It's almost too bad, it was a neat trick.
    with style.latex(width, 1):
        style.pretty()
        plt.figure(1)
        plt.subplot(2, 1, 1)
        plt.plot(x_folded, y_folded_h, color=style.COLORS_STD[0], label=style.latexT("H"))
        plt.ylim((8.7, 8.9))
        plt.title(style.latexT("LO + sky, folded."), fontsize=10)
        plt.ylabel(style.latexT("Flux density  [\\si{\watt\per\meter\squared}]"))
        plt.legend(prop={"size":6})

        plt.subplot(2, 1, 2)
        plt.plot(x_folded, y_folded_v, color=style.COLORS_STD[0], label=style.latexT("V"))
        plt.ylim((2.6, 2.8))
        plt.xlabel(style.latexT("Intermediate frequency [\\si{\giga\hertz}]"))
        plt.ylabel(style.latexT("Flux density  [\\si{\watt\per\meter\squared}]"))
        plt.legend(prop={"size":6})

        if USE_PDF:
            plt.savefig("thin_film_beam_splitter_folded.pdf", bbox_inches='tight')
        else:
            plt.show()

    # Compute FFT.
    n = len(x_folded)  # Number of samples.
    h = np.hanning(n)
    y_fft_h = np.abs(np.fft.rfft(y_folded_h * h))
    y_fft_v = np.abs(np.fft.rfft(y_folded_v * h))
    w = (np.max(x_folded) - np.min(x_folded)) * 1e9  # Width.
#     n = len(xs[usb])
#     h = np.hanning(n)
#     y_fft_h = np.abs(np.fft.rfft(lo_h[usb]))  # * h))
#     y_fft_v = np.abs(np.fft.rfft(lo_v[usb]))  # * h))
#     w = (np.max(xs[usb]) - np.min(xs[usb])) * 1e9  # Width.
    s = w / n  # Sample spacing.
    x_fft = np.fft.fftfreq(n, s)[:len(y_fft_h)]  # Only positive frequencies.
    x_fft = 1 / x_fft / 1e6  # x scale in SW period.

    # Plot FFT.

    # What would it take to see the 10 micrometer beam splitter?
    # d_0 = 1.9999995  m
    # d_1 = 1.0000005  m
    # T_0 = c / 2d_0  Hz
    # T_1 = c / 2d_1  Hz
    # F_0 = 2d_0 / c  Hz-1  # This is the frequency of the FFT, in Hz-1.
    # F_1 = 2d_1 / c  Hz-1
    # \Delta F = F_1 - F_0 = 2(0.000001)/c  Hz-1
    # \Delta F = 1 / B  # With B the bandwith.
    # 1 / B = 0.000002 / c
    # B = c / 0.000002  =  1.5e14  Hz
    # Well, I will not be having such a bandwidth any time soon so the two peaks
    # due to the two distances will not show.
    # Our B equals 4 GHz.
    # c / B = 0.075 m.
    # 2d = 0.075 ; d = 0.0375
    # With 4 GHz of bandwidth I should expect to resolve differences of 3.7 mm.
    # So if I make the beam splitter 1 cm thick, I should be able to see it on the FFT
    # as a second peak.
    # Well, it does not work.  Even if I use 10 cm.  Even if I remove the Hanning window.
    # The peak does move to the higher periods, but I do not see a second peak.
    # Maybe it is much weaker?  Use semilogy.  Still not.
    # Maybe I should try on the non-folded ?  Still not.
    # Maybe I am wrong to think that I should see two peaks?

    # The thicker the film, the greater the period of the single peak.  This
    # means that the cavity seems shorter.  Is that at least consistent with the
    # fact that the speed of light is smaller in the film?
    # The spacing between the resonant frequencies is proportional to c:
    #     f = N c / 2d, with N an integer.
    # The period is c / 2d: distance between two resonant frequencies.

    # So a slower light should reduce the period, not increase it.  Indeed,
    # reducing the speed is kinda like increasing the distance. Then why does my
    # peak move to higher periods as the thickness of the film increase? Higher
    # periods suggests that the distance is reduced.

    # Well, there is one distance that is reduced when the thickness increases,
    # that is the distance to the near side.  Does it match?  Standard is a 1 m
    # cavity, creating a period of 150 MHz.  If I make it a 90 cm cavity I
    # should get 166.6667 MHz.

    # The film must have a thickness h which, when seen at 45 degrees, appear to
    # be l=10 cm long.  h and l are linked by a cosine and h is smaller than l,
    # so I say h = l cos 45, h = 1/sqrt(2) / 10.

    # Once the model ran, I get a peak at 166.6.  The resolution is not high, but
    # it is high enough to tell me I hit at the right first decimal.

    # So what we mostly see is a reflection on the near side.  What about the
    # far side? How come it has so little effect?  Especially because the film
    # is mostly transparent, so there should be quite a bit of power hitting the
    # far side.  Then a bit of that power comes to the near side and is transmitted.
    # Because the transmissions are close to 1, I'd expect that the power in the near
    # and the far contributions to be very close.

    # SOLVED IT !  My problem was that my film had an absorption coefficient.
    # That small imaginary component to n2 does not matter much for a 10 micrometer
    # film, but it totally killed all transmission in a 10 cm film.  As a result, all
    # I was seeing was the near side, and 10^-14 of the far side.  If I want to see all
    # the cavities, I must make n2 a real number.  Then I see many peaks, not just 2,
    # because each double reflection brings its own cavity length.

    with style.latex(width, 1):
        style.pretty()
        plt.figure(2)
        plt.subplot(2, 1, 1)
        plt.semilogy(x_fft[10:], y_fft_h[10:], color=style.COLORS_STD[0], label=style.latexT("H"))
        plt.title(style.latexT("LO + sky, folded."), fontsize=10)
        plt.ylabel(style.latexT("FFT of flux density  [arbitrary]"))
#         plt.xlim((50, 250))
        plt.legend(prop={"size":6})

        plt.subplot(2, 1, 2)
        plt.semilogy(x_fft[10:], y_fft_v[10:], color=style.COLORS_STD[0], label=style.latexT("V"))
        plt.xlabel(style.latexT("Period [\\si{\mega\hertz}]"))
        plt.ylabel(style.latexT("FFT of flux density  [arbitrary]"))
#         plt.xlim((50, 250))
        plt.legend(prop={"size":6})
        if USE_PDF:
            plt.savefig("thin_film_beam_splitter_folded_fft.pdf", bbox_inches='tight')
        else:
            plt.show()



def main(width):
    frequencies, b_lo, b_sky = compute()
    plot(width, frequencies, b_lo, b_sky)

if __name__ == '__main__':
    if False:
        import cProfile as profile
        profile.run('compute()')
    else:
        main(style.WIDTH_ARTICLE)
