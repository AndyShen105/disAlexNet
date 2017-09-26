#!/bin/bash
#$1:the start index of cluster eg；ssd32-32
#$2:the end index of cluster 
#$3:target port

#global variable
pid=""

get_pid(){
#get pid of port
#$1 ip
#$2 target port
#$3 pid
re=`ssh $1 "netstat -anp|grep "$2`
tem=${re%%/*}
pid=${tem##* }
if [ "$pid" != "" ]
then
    ssh $1 kill $pid
fi
}

for((step=$1;step<=$2;step++));
do
    if [ $step == 42 ]
    then
	re=`netstat -anp|grep $3`
	tem=${re%%/*}
	pid=${tem##* }
	if [ "$pid" != "" ]
	then
	    kill $pid
	fi
    else
	get_pid "ssd"$step $3
    fi
done
echo "clean up"
