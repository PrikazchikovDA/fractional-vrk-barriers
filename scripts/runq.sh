#!/bin/sh
cd "$(dirname "$0")"
while [ -n "$(ps -eo args | grep -E '^python3 s5_cb.py (0.2 2.6|0.25 2.75)')" ]; do sleep 20; done
touch done_jobs.txt
grep -vxFf done_jobs.txt jobs_rest.txt | xargs -P 2 -I{} sh -c '{} && echo "{}" >> done_jobs.txt'
echo QUEUE_FINISHED
