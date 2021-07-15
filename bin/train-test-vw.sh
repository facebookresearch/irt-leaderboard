# Copyright (c) Facebook, Inc. and its affiliates.
# hyper parames from:
# python scripts/vw-hyperopt.py --train /tmp/feature-irt/train.vw.txt --holdout /tmp/feature-irt/test.vw.txt --vw_space '--learning_rate=.001..10~L --loss_function=logistic --link=logistic --passes=10 --l2=1e-8..1e-1~L0 -b 20' --plot

mkdir -p /tmp/feature-irt/plots/

vw  -d /tmp/feature-irt/train.vw.txt -f /tmp/model.vs --holdout_off -c  --l2 1.0009034388669644e-08 --learning_rate 0.9379337862402656 --link logistic --loss_function logistic --passes 10 -b 24

vw --testonly -d /tmp/feature-irt/test.vw.txt -i /tmp/model.vw -p /tmp/feature-irt/test.pred.txt -k --loss_function logistic --link logistic
leaderboard features eval-vw /tmp/feature-irt/test.vw.txt /tmp/feature-irt/test.pred.txt /tmp/feature-irt/plots