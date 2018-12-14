cython bbox.pyx
cython cython_nms.pyx
# cython gpu_nms.pyx
# python setup_cpu.py build_ext --inplace
python mysetup.py build_ext
mv utils/* ./
rm -rf build
rm -rf utils

