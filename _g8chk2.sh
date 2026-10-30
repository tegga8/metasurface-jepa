#!/bin/bash
# Full tail of the gate-8 log (the runs produced no JSON -> they failed).
tail -40 "$HOME/logs/gate8_642386.out" 2>/dev/null
echo "G8CHK2_DONE"
