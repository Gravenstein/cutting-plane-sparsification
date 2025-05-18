#!/usr/bin/sh
CSVFILE=$1
SCIPBIN=$2
RESULTDIR=$3

TIMELIMIT=39600

LINENO=$SLURM_ARRAY_TASK_ID
LINE=$(head $CSVFILE -n $LINENO | tail -n 1 | cut -d , -f 2)
PROBNAME=$(head $CSVFILE -n $LINENO | tail -n 1 | cut -d , -f 1)
PROBFILE=$(head $CSVFILE -n $LINENO | tail -n 1 | cut -d , -f 2)
SOLFILE=$RESULTDIR/$PROBNAME.sol.gz
ls $SOLFILE
if [ $? -eq 0 ]
then
    echo $SOLFILE already exists.
    exit 0
fi
$SCIPBIN -c "set limits time $TIMELIMIT" -c "read $PROBFILE" -c "opt" -c "write solution $RESULTDIR/$PROBNAME.sol.gz" -c "quit"
