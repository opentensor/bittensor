
#!/bin/bash
counter=1
while [ $counter -le 10000 ]
do
    echo "python3 ./bin/btcli register --wallet.name sud --no_prompt --netuid 0 --wallet.hotkey $counter"
    python3 ./bin/btcli register --wallet.name sud --no_prompt --netuid 0 --wallet.hotkey $counter
    ((counter++))
done