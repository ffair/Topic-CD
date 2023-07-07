experiment_dir="experiments/cell"
pass=seed18
nohup python dphmm.py --experiment_dir ${experiment_dir} --prefix ${pass} > ${experiment_dir}/${pass}_logs.txt 2>&1 &
