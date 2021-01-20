mkdir -p ~/.bittensor/wallets/default/hotkeys
rm ~/.bittensor/wallets/default/coldkeypub.txt
rm ~/.bittensor/wallets/default/hotkeys/default
touch ~/.bittensor/wallets/default/coldkeypub.txt
touch ~/.bittensor/wallets/default/hotkeys/default
echo "0x74acaa8d7829336dfff7569f19225818cc593335b9aafcde3f69db23c3538561" >> ~/.bittensor/wallets/default/coldkeypub.txt
echo '{"accountId": "0x9cf7085aa3304c21dc0f571c0134abb12f2e8e1bc9dbfc82440b8d6ba7908655", "publicKey": "0x9cf7085aa3304c21dc0f571c0134abb12f2e8e1bc9dbfc82440b8d6ba7908655", "secretPhrase": "document usage siren cross across crater shrug jump marine distance absurd caught", "secretSeed": "0x2465ae0757117bea271ad622e1cd0c4b319c96896a3c7d9469a68e63cf7f9646", "ss58Address": "5FcWiCiFoSspGGocSxzatNL5kT6cjxjXQ9LuAuYbvFNUqcfX"}' >> ~/.bittensor/wallets/default/hotkeys/default
chmod 0600 ~/.bittensor/wallets/default/coldkeypub.txt
chmod 0600 ~/.bittensor/wallets/default/hotkeys/default
echo "~/.bittensor/wallets/default/coldkeypub.txt"
cat ~/.bittensor/wallets/default/coldkeypub.txt
echo "~/.bittensor/wallets/default/hotkeys/default"
cat ~/.bittensor/wallets/default/hotkeys/default