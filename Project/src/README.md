# Compile and run
Remember to load CUDA on the system path "PATH" and "LD_LIBRARY_PATH".

```bash
cd src
mkdir build
cd build
cmake -G Ninja -DCMAKE_CXX_COMPILER=nvcc ..
cmake --build .
./hs --video=../../dataset/schoolgirls.mp4
```

