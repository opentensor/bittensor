#!/bin/bash
set -u

abort() {
  printf "%s\n" "$1"
  exit 1
}

getc() {
  local save_state
  save_state=$(/bin/stty -g)
  /bin/stty raw -echo
  IFS= read -r -n 1 -d '' "$@"
  /bin/stty "$save_state"
}

wait_for_user() {
  local c
  echo
  echo "Press RETURN to continue or any other key to abort"
  getc c
  # we test for \r and \n because some stuff does \r instead
  if ! [[ "$c" == $'\r' || "$c" == $'\n' ]]; then
    exit 1
  fi
}

shell_join() {
  local arg
  printf "%s" "$1"
  shift
  for arg in "$@"; do
    printf " "
    printf "%s" "${arg// /\ }"
  done
}

# string formatters
if [[ -t 1 ]]; then
  tty_escape() { printf "\033[%sm" "$1"; }
else
  tty_escape() { :; }
fi
tty_mkbold() { tty_escape "1;$1"; }
tty_underline="$(tty_escape "4;39")"
tty_blue="$(tty_mkbold 34)"
tty_red="$(tty_mkbold 31)"
tty_bold="$(tty_mkbold 39)"
tty_reset="$(tty_escape 0)"

ohai() {
  printf "${tty_blue}==>${tty_bold} %s${tty_reset}\n" "$(shell_join "$@")"
}

# Things can fail later if `pwd` doesn't exist.
# Also sudo prints a warning message for no good reason
cd "/usr" || exit 1

linux_install_pre() {
    sudo apt-get update 
    sudo apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential 
}

linux_install_python() {
    which -s python3.8
    if [[ $? != 0 ]] ; then
        ohai "Installing python3.8"
        sudo apt-get install --no-install-recommends --no-install-suggests -y python3.8
    else
        ohai "Updating python3.8"
        sudo apt-get update python3.8
    fi
    ohai "Installing python tools"
    sudo apt-get install --no-install-recommends --no-install-suggests -y python3-pip python3.8-dev python3.8-venv
}

linux_activate_installed_python() {
    ohai "Creating python virtualenv"
    mkdir -p ~/.bittensor/bittensor
    cd ~/.bittensor/
    python3.8 -m venv env
    ohai "Entering bittensor-environment"
    source env/bin/activate
    ohai "You are using python@3.8$"
    ohai "Installing python tools"
    python -m pip install --upgrade pip
    python -m pip install python-dev
}

linux_install_bittensor() {
    ohai "Cloning bittensor@master into ~/.bittensor/bittensor"
    mkdir -p ~/.bittensor/bittensor
    git clone https://github.com/opentensor/bittensor.git ~/.bittensor/bittensor/ 2> /dev/null || (cd ~/.bittensor/bittensor/ ; git pull --ff-only)
    ohai "Installing bittensor"
    python -m pip install -e ~/.bittensor/bittensor/
}


mac_install_xcode() {
    which -s xcode-select
    if [[ $? != 0 ]] ; then
        ohai "Installing xcode:"
        xcode-select --install
    fi
}

mac_install_brew() {
    which -s brew
    if [[ $? != 0 ]] ; then
        ohai "Installing brew:"
        ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    else
        ohai "Updating brew:"
        brew update --verbose
    fi
}

mac_install_cmake() {
    which -s cmake
    if [[ $? != 0 ]] ; then
        ohai "Installing cmake:"
        brew install cmake
    else
        ohai "Updating cmake:"
        brew upgrade cmake
    fi
}

mac_install_python() {
    which -s python3.7
    ohai "Installing python3.7"
    brew list python@3.7 &>/dev/null || brew install python@3.7;
    ohai "Updating python3.7"
    brew upgrade python@3.7
}

mac_activate_installed_python() {
    ohai "Creating python virtualenv"
    mkdir -p ~/.bittensor/bittensor
    cd ~/.bittensor/
    /usr/local/opt/python@3.7/bin/python3 -m venv env
    ohai "Entering python3.7 environment"
    source env/bin/activate
    PYTHONPATH=$(which python)
    ohai "You are using python@ $PYTHONPATH$"
    ohai "Installing python tools"
    python -m pip install --upgrade pip
    python -m pip install python-dev
}

mac_install_bittensor() {
    ohai "Cloning bittensor@master into ~/.bittensor/bittensor"
    git clone https://github.com/opentensor/bittensor.git ~/.bittensor/bittensor/ 2> /dev/null || (cd ~/.bittensor/bittensor/ ; git pull --ff-only)
    ohai "Installing bittensor"
    python -m pip install -e ~/.bittensor/bittensor/
    deactivate
}

# Do install.
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then

    which -s apt
    if [[ $? == 0 ]] ; then
        abort "This linux based install requires apt. To run with other distros (centos, arch, etc), you will need to manually install the requirements"
    fi

    ohai "This script will install:"
    echo "git"
    echo "curl"
    echo "git"
    echo "cmake"
    echo "build-essential"
    echo "python3.8"
    echo "python3.8-pip"
    echo "bittensor"

    wait_for_user
    linux_install_pre
    linux_install_python
    linux_activate_installed_python
    linux_install_bittensor

elif [[ "$OS" == "Darwin" ]]; then
    ohai "This script will install:"
    echo "xcode"
    echo "homebrew"
    echo "git"
    echo "cmake"
    echo "python3.7"
    echo "python3.7-pip"
    echo "bittensor"

    wait_for_user
    mac_install_brew
    mac_install_cmake
    mac_install_python
    mac_activate_installed_python
    mac_install_bittensor

else
  abort "Bittensor is only supported on macOS and Linux"
fi

# Use the shell's audible bell.
if [[ -t 1 ]]; then
printf "\a"
fi
ohai "Installation successful!"
echo ""
ohai "Next steps:"
echo ""
echo "- 1) Choose your network: "
echo "    $ export NETWORK=akira      # Test network (suggested)" 
echo "    $ export NETWORK=kusanagi   # Main network (production)"
echo ""
echo "- 2) Activate the installed python: "
echo "    $ source ~/.bittensor/env/bin/activate"
echo ""
echo "- 3) Create a wallet: "
echo "    $ export WALLET=<your wallet name>"
echo "    $ export HOTKEY=<your hotkey name>"
echo "    $ bittensor-cli new_coldkey --wallet.name \$WALLET"
echo "    $ bittensor-cli new_hotkey --wallet.name \$WALLET --wallet.hotkey \$HOTKEY"
echo ""
echo "- 4) (Optional) Open a port on your NAT: "
echo "    See Docs: ${tty_underline}https://opentensor.github.io/index.html${tty_reset})"
echo ""
echo "- 5) Run a miner: "
echo "    i.e. $ python ~/.bittensor/bittensor/miners/TEXT/gpt2_wiki.py"
echo "                   --subtensor.network \$NETWORK"       
echo "                   --wallet.name \$WALLET"
echo "                   --wallet.hotkey \$HOTKEY"
echo ""
ohai "Extras:"
echo ""
echo "- Read the docs: "
echo "    ${tty_underline}https://opentensor.github.io/index.html${tty_reset}"
echo ""
echo "- Visit our website: "
echo "    ${tty_underline}https://www.bittensor.com${tty_reset}"
echo ""
echo "- Join the discussion: "
echo "    ${tty_underline}https://discord.gg/3rUr6EcvbB${tty_reset}"
echo ""

    
