### Usage

Version 2 is designed to maximize ease of use: it will probably suffice to just call one function, namely

```
direct_sht.points2alm_host(theta, phi, wt, lmax)
```

where theta, phi, wt are 1-d real-valued numpy arrays of the same length (each array element represents one galaxy). This function will automatically distribute the points to all GPUs in the node, take the direct SHT, and return the result as a complex-valued alm array in healpy ordering.

There are also lower-level functions to manage arrays on individual GPUs, and integrate with cupy, but I suspect you won't need them! If you think they might be useful, let me know and I can provide some guidance.

You might also find it useful to look at the source code for `direct_sht.tests.compare_to_healpy()`, which verifies that healpy.map2alm() and direct_sht.points2alm() give the same result, in the special case of a "catalog" with one entry per pixel. (This test is run automatically by `python -m direct_sht test`, see below):

https://github.com/kmsmith137/direct_sht/blob/main/direct_sht/tests.py

### Installation

1. Create a conda env. This works on perlmutter:
```
# Create and activate a conda env named 'direct_sht'.
# Note: Only need to do 'conda create' once.

module load conda
conda create -n direct_sht cupy pybind11 healpy scipy
conda activate direct_sht
```

2. Install the `ksgpu` library (https://github.com/kmsmith137/ksgpu).

3. Install `direct_sht`. The build system supports either python builds with `pip`,
or C++ builds with `make`. Here's what I recommend:
```
    # Step 1. Clone the repo and build with 'make', so that you can read
    # the error messages if anything goes wrong. (pip either generates too
    # little output or too much output, depending on whether you use -v).

    git clone https://github.com/kmsmith137/direct_sht
    cd direct_sht
    make -j 32

    # Step 2: Run some unit tests, just to check that it worked
    # (incl. comparison to healpy).

    python -m direct_sht test

    # Step 3 (optional): Run some timings. Suggest using a compute node.

    python -m direct_sht time

    # Step 4 (optional): If everything looks good, build an editable pip install.
    # This will let you import 'direct_sht' outside the build dir.
    # This only needs to be done once per conda env (or virtualenv).
    
    pip install pipmake
    pip install --no-build-isolation -v -e .    # -e for "editable" install

    # Step 5: In the future, if you want to rebuild direct_sht (e.g. after a
    # git pull), you can ignore pip and build with 'make'. (This is only
    # true for editable installs -- for a non-editable install you need
    # to do 'pip install' again.)

    git pull
    make -j 32   # no pip install needed, if existing install is editable
```
Let me know if you have any trouble compiling! (I'm hoping soon to put some precompiled wheels on pypi so that you can pip install without compiling, but I didn't quite get to it in this release.)
