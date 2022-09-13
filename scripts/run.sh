#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

n_process=2
PY=python3.7
vars="3 5 8 10"
objs="2 3"
seeds="1"
algos="DMI ParEGO MoeadEGO"

pros_2D="ZDT3 ZDT31  ZDT32  ZDT33  ZDT34  WFG21  WFG22  WFG23  WFG24"
pros_3D="DTLZ7  DTLZ71  DTLZ72 DTLZ73 WFG2  WFG21  WFG22  WFG23  WFG24"
n_weight_2D=100
n_weight_3D=120
RUNNAME=vmware-vmx

pids=()
p=1
active_proc=0

wait_empty_processor(){
  echo "[SHELL] Wait empty processor $active_proc/$n_process"
  while [ $active_proc -ge $n_process ]
  do
    for j in $( seq 1 ${#pids[@]} )
    do
      #echo "check $j"
      if [ -z "${pids[$j]}" ]
      then
        echo "we have empty pids[${j}]"
      else
        if [ "${pids[$j]}" -ne -1 ]
        then
          if [ -z "`ps aux | awk '{print $2 }' | grep ${pids[$j]}`" ]
          then
            echo "[SHELL] $j:${pids[$j]} Finish $(date +"%T")"
            pids[$j]=-1
            let active_proc=$active_proc-1

          fi
        fi
      fi
    done
    sleep 120
  done
}

rm -rf ./$RUNNAME
ln -s `which $PY` ./$RUNNAME
for s in $seeds
do
  for var in $vars
  do
    for obj in $objs
    do
      pros="pros_${obj}D"
      n_weight="n_weight_${obj}D"
      for pro in ${!pros}
      do
        for algo in ${algos}
        do
            let init=$var*11-1
            let iter=$var*$var*11-5
            cmd="./$RUNNAME main.py --algorithm $algo --seed $s --n-var $var --n-obj $obj --n-init $init  --n-iter $iter --problem-name $pro --n-weight=${!n_weight} >output/log/"
            echo $cmd
            $cmd &
            pids[$p]=$!
            echo "[SHELL]pids[${p}]=${pids[$p]} :$cmd Start $(date +"%T")"
            let p=$p+1
            let active_proc=$active_proc+1
            sleep 10
            wait_empty_processor
        done
      done
    done
  done
done
echo "[SHELL] All runs added!"
