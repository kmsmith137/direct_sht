Installation (requires CUDA and an NVIDIA A30/A100 GPU)

```
# First, build 'gputils' library (dependency)
git clone https://github.com/kmsmith137/gputils
cd gputils
make -j
cd ..

git clone https://github.com/kmsmith137/direct_sht
cd direct_sht
make -j

# Run some tests
./bin/test-sht

# Run some timings
./bin/time-sht
```