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

linux_install_pre() {
    sudo apt-get update 
    sudo apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip 
}

linux_install_python() {
    which -s python3.7
    if [[ $? != 0 ]] ; then
        ohai "Installing python3.7"
        sudo apt-get install --no-install-recommends --no-install-suggests -y python3.7 python-pip python3-dev   
    else
        ohai "Updating python3.7"
        sudo apt-get update python3.7
    fi
    ohai "Installing bittensor deps"
    python3.7 -m pip install python-dev
    python3.7 -m pip install --upgrade pip
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
        brew update
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
    if [[ $? != 0 ]] ; then
        ohai "Installing python3.7"
        brew install python@3.7
    else
        ohai "Updating python3.7"
        brew upgrade python@3.7
    fi
    ohai "Installing bittensor deps"
    python3.7 -m pip install python-dev
    python3.7 -m pip install --upgrade pip
}

install_bittensor() {
    mkdir tmp
    cd tmp
    ohai "Cloning bittensor"
    git clone https://github.com/opentensor/bittensor.git
    cd bittensor
    ohai "Installing bittensor"
    python3.7 -m pip install -e .
    cd ../..
    rm -rf tmp
}

# Do install.
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
    ohai "This script will install:"
    echo "git"
    echo "curl"
    echo "git"
    echo "cmake"
    echo "build-essential"
    echo "python3.7"
    echo "python3.7-pip"

    wait_for_user
    linux_install_pre
    linux_install_python
    install_bittensor
    ohai "Installation successful!"

elif [[ "$OS" == "Darwin" ]]; then
    ohai "This script will install:"
    echo "xcode"
    echo "homebrew"
    echo "git"
    echo "cmake"
    echo "python3.7"
    echo "python3.7-pip"

    wait_for_user
    mac_install_brew
    mac_install_cmake
    mac_install_python
    install_bittensor
    ohai "Installation successful!"
    

else
  abort "Bittensor is only supported on macOS and Linux."

fi

# Use the shell's audible bell.
if [[ -t 1 ]]; then
  printf "\a"
fi
ohai "Next steps:"
echo "- First \`bittensor-cli new_wallet\` to create a new wallet "
echo "- Second \`python7 bittensor/examples/...\` to run a miner "

echo "- Read the docs: "
echo "    ${tty_underline}https://opentensor.github.io/index.html${tty_reset}"
echo "- Visit our website: "
echo "    ${tty_underline}https://www.bittensor.com${tty_reset}"
echo "- Join the discussion: "
echo "    ${tty_underline}https://discord.gg/3rUr6EcvbB${tty_reset}"




