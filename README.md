### Usage

Version 2 is designed to maximize ease of use: it will probably suffice to just call one function, namely

```
direct_sht.points2alm_host(theta, phi, wt, lmax)
```

where theta, phi, wt are 1-d real-valued numpy arrays of the same length (each array element represents one galaxy). This function will automatically distribute the points to all GPUs in the node, take the direct SHT, and return the result as a complex-valued alm array in healpy ordering.

There are also lower-level functions to manage arrays on individual GPUs, and integrate with cupy, but I suspect you won't need them! If you think they might be useful, let me know and I can provide some guidance.

You might also find it useful to look at the source code for `direct_sht.tests.compare_to_healpy()`, which verifies that healpy.map2alm() and direct_sht.points2alm() give the same result, in the special case of a "catalog" with one entry per pixel. (This test is run automatically by `python -m direct_sht test`, see below):

https://github.com/kmsmith137/direct_sht/blob/main/src_python/direct_sht/tests.py

### Installation

Currently installation is awkward, sorry about that! I haven't quite resolved some nuisance issues with 'pip install', so you'll need to use weird 'pip install' flags (see below). Also, you need my helper library 'gputils', but you'll need to use the `python` branch, not the main branch. (I had to make some changes to this library, but can't merge to the main branch just yet without screwing up other projects.)

I got a nersc account (thanks to SO membership) and verified that the following works on perlmutter:

```
# Create and activate a conda env named 'direct_sht'.
# Note: Only need to do 'conda create' once.
# Note: healpy doesn't work with the latest scipy (1.14), 
# so I had to use scipy 1.13.

module load conda
conda create -n direct_sht cupy meson-python \
   pybind11 healpy scipy==1.13  # note scipy version number
conda activate direct_sht

# Compile gputils library (note -b flag to select 'python' branch)
git clone https://github.com/kmsmith137/gputils -b python   
cd gputils
pip install --no-cache-dir --no-build-isolation -v .   # note weird pip flags
cd ..

# Compile direct_sht library
git clone https://github.com/kmsmith137/direct_sht
cd direct_sht
pip install --no-cache-dir --no-build-isolation -v .    # note weird pip flags
cd ..

# Run some unit tests (incl. comparison to healpy)
python -m direct_sht test

# Run some timings (suggest using compute node)
python -m direct_sht time 
```

Let me know if you have any trouble compiling! (I'm hoping soon to put some precompiled wheels on pypi so that you can pip install without compiling, but I didn't quite get to it in this release.)
