PY=python3.7

cd problems/clib
rm -rf *.so
clib_list="ZDT DTLZ WFG IDTLZ MDTLZ"
for clib in $clib_list
do
  echo "$clib/setup.py build"
  $PY $clib/setup.py build
  mv build/lib*/*.so ./lib$clib.so
  rm -rf build/
done