#!/bin/bash

#for i in $(seq 0 4)
#do
#    mod=$(python paramGetter.py model /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
#    ma=$(python paramGetter.py args /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
#    mk=$(python paramGetter.py kwargs /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
#    printf "\n\nFront\tModel M$i\n"
#    python main.py --train-dataset none --test-dataset fbirn --test-dataset-kwargs '{}' --inference-model "/home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/checkpoints/best.pth" --model $mod --model-args $ma --model-kwargs $mk --logdir "logs/fbirn/Inference_Example_M_${i}_0" --inference-only 1
#done

for i in $(seq 0 4)
do
    mod=$(python paramGetter.py model /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
    ma=$(python paramGetter.py args /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
    mk=$(python paramGetter.py kwargs /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
    printf "\n\nFront\tModel M$i\n"
    python main.py --train-dataset none --test-dataset mddc --test-dataset-kwargs '{}' --inference-model "/home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/checkpoints/best.pth" --model $mod --model-args $ma --model-kwargs $mk --logdir "logs/mdd/Inference_Example_M_${i}_0" --inference-only 1
done

#for i in $(seq 0 4)
#do
#    mod=$(python paramGetter.py model /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
#    ma=$(python paramGetter.py args /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
#    mk=$(python paramGetter.py kwargs /home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/)
#    printf "\n\nFront\tModel M$i\n"
#    python main.py --train-dataset none --test-dataset ptsd --test-dataset-kwargs '{}' --inference-model "/home/users/sedwardsswart/Documents/DFNC/Models/M003_canonical_lstm_undersampled/run_$i/checkpoints/best.pth" --model $mod --model-args $ma --model-kwargs $mk --logdir "logs/ptsd/Inference_Example_M_${i}_0" --inference-only 1
#done
