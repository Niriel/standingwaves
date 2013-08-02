p = obs[0].product.refs["level2"].product.refs["WBS-H-USB"].product.refs["box_001"].product
keys = p.keySet()
maxes = []
baselines = []
noises = []
for key in keys:
    ds = p[key]
    m = -1000000
    b = 1000000
    n = 1000000
    for sb in range(1, 5):
        flux = ds.getFlux(sb)
        m = max(m, MAX(flux))
        b = min(m, MEDIAN(flux))
        print key, sb, MAX(flux), STDDEV(flux)
        n = min(n, STDDEV(flux))
    maxes.append(m)
    baselines.append(b)
    noises.append(n)
maxes = Double1d(maxes)
baselines = Double1d(baselines)
noises = Double1d(noises)

maxes = maxes[2:]
baselines = baselines[2:]
noises = noises[2:]
assert len(maxes) == len(noises)
print STDDEV(maxes-baselines)
print MEDIAN(noises)
print STDDEV(maxes-baselines) / MEDIAN(noises)
