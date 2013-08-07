QUERY_RESULT = ProductStorage([PoolManager.getPool('hifi-aux'), PoolManager.getPool('pipeline-out'), PoolManager.getPool('standard'), PoolManager.getPool('hifi-cal'), PoolManager.getPool('hsa')]).select(herschel.ia.pal.query.MetaQuery(herschel.ia.obs.ObservationContext, "p", "(p.meta.containsKey(\"obsid\") and p.meta[\"obsid\"].value==1342190892L)"))
obs = QUERY_RESULT[0]

p = obs.product.refs["level2"].product.refs["WBS-H-USB"].product.refs["box_001"].product
keys = p.keySet()

for key in keys:
    ds = p[key]
    lo = ds.meta["loFrequency"].value
    print lo
    xs = []
    ys = []
    for sb_id in range(1, 5):
        x = ds.getWave(sb_id)
        y = ds.getFlux(sb_id)
        assert len(x) == len(y)
        xs.extend(x)
        ys.extend(y)
    xss = map(str, xs)
    yss = map(str, ys)
    line_x = " ".join(xss) + "\n"
    line_y = " ".join(yss) + "\n"
    filename = "%09.4f.csv" % lo
    f = open(filename, 'w')
    f.writelines([line_x, line_y])
    f.close()

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
