import sys

if (len(sys.argv) == 2) and (sys.argv[1] == 'test'):
    from . import tests
    tests.multi_compare_to_healpy()
elif (len(sys.argv) == 2) and (sys.argv[1] == 'time'):
    from . import time_direct_sht
    time_direct_sht.time_points2alm(npoints=1000*1000, lmax=1000)
else:
    print(f'Usage: {sys.argv[0]} [test | time]')
