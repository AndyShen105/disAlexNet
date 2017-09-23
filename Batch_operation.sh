#!/bin/bash
#$1 id_start eg:ssd35's id is 35
#$2 id_end
#$3 cm
echo $1
echo $2
for(( i=$1; i < $2; i++ ))
do
    ssh ssd$i $3
    echo $x
done
    

