#!/bin/bash
wip=~/wip

for i in $*; do
    name=${i//.log/}
    echo $name
    python $wip/ort-web-profile.py $i
    # echo "python $wip/ort-trace.py --provider --dtypes -l 50 $name"".json --webgpu $name""_gpu.json --exclude If"
    if [ -r "$name"_gpu.json ]; then
        opt="--webgpu ${name}_gpu.json"
    fi
    python $wip/ort-trace-color.py --input $name.json --output $name-opt.json $opt
    python $wip/ort-trace.py --provider --type-in-name -l 50 $name.json $opt --exclude If > $name.txt
    python $wip/ort-trace.py --provider --type-in-name -l 50 $name.json --shapes $opt --exclude If > $name-shapes.txt
    python $wip/ort-trace.py $name.json --dtypes --provider -l 500 --cov $wip/ort-perf-log/webgpu.txt --csv $name.csv
    python $wip/ort-trace-flow.py $name.json --nodes > ${name}_flow.txt
    # python $wip/ort-trace-flow.py $name.json > ${name}_flow1.txt
done


rm -f all.csv
cat *.csv > all.csv
head -1 all.csv > webgpu-ops-coverage.csv
grep -v op_type all.csv >> webgpu-ops-coverage.csv
