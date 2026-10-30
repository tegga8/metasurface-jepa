#!/bin/bash
# Just the queue.
squeue -u mm24r002 -o "%.7i %.9j %.2t %.9M"
echo Q_DONE
