#!/usr/bin/env bash

#SBATCH --job-name=leaderboard
#SBATCH --time=1-00:00:00
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --chdir=/fs/clip-quiz/entilzha/code/leaderboard
#SBATCH --output=/fs/clip-quiz/entilzha/logs/%A.log
#SBATCH --error=/fs/clip-quiz/entilzha/logs/%A.log
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8g
#SBATCH --exclude=chroneme03,clipgpu00,clipgpu01,clipgpu02,clipgpu03,materialgpu00,materialgpu01,materialgpu02

# Copyright (c) Facebook, Inc. and its affiliates.
set -x
hostname
source /fs/clip-quiz/entilzha/anaconda3/etc/profile.d/conda.sh > /dev/null 2> /dev/null
conda activate leaderboard

srun leaderboard rank_stability run $1 $2 $3