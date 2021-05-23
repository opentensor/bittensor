#!/usr/bin/env bash


count=0

_stopnow() {
    count="$(($count+1))"
    test -f stopnow && \
      echo "Stopping after $count iterations!" && \
      rm stopnow && exit 0 || return 0
}

control_c()
# run if user hits control-c
{
  echo "Managed to do $count iterations"
  exit $?
}

#trap keyboard interrupt (control-c)
trap control_c SIGINT

echo "To stop this forever loop created a file called stopnow."
echo "E.g: touch stopnow"
echo ""
echo "Now going to run '$@' forever"
echo ""
while true
do
    _stopnow

    eval $@

    # Do this in case you accidentally pass an argument
    # that finishes too quickly.
    sleep 1

done
