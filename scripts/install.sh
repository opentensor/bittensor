
#!/bin/bash
set -u

python="python3"

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
    which $python
    if [[ $? != 0 ]] ; then
        ohai "Installing python"
        sudo apt-get install --no-install-recommends --no-install-suggests -y python3.8
    else
        ohai "Updating python"
        sudo apt-get update python3
    fi
    ohai "Installing python tools"
    sudo apt-get install --no-install-recommends --no-install-suggests -y python3-pip python3-dev 
}

linux_update_pip() {
    PYTHONPATH=$(which python)
    ohai "You are using python@ $PYTHONPATH$"
    ohai "Installing python tools"
    $python -m pip install --upgrade pip
}

linux_install_bittensor() {
    ohai "Cloning bittensor@master into ~/.bittensor/bittensor"
    mkdir -p ~/.bittensor/bittensor
    git clone https://github.com/opentensor/bittensor.git ~/.bittensor/bittensor/ 2> /dev/null || (cd ~/.bittensor/bittensor/ ; git pull --ff-only)
    ohai "Installing bittensor"
    $python -m pip install -e ~/.bittensor/bittensor/
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

mac_update_pip() {
    PYTHONPATH=$(which python)
    ohai "You are using python@ $PYTHONPATH$"
    ohai "Installing python tools"
    $python -m pip install --upgrade pip
}

mac_install_bittensor() {
    ohai "Cloning bittensor@master into ~/.bittensor/bittensor"
    git clone https://github.com/opentensor/bittensor.git ~/.bittensor/bittensor/ 2> /dev/null || (cd ~/.bittensor/bittensor/ ; git pull --ff-only)
    ohai "Installing bittensor"
    $python -m pip install -e ~/.bittensor/bittensor/
    deactivate
}

setup_wallet_and_miner() {
    echo ""
    echo ""
    while true
    do
      read -r -p "Do you wish to create a Bittensor wallet? [Y/n] " input
    
      case $input in
          [yY][eE][sS]|[yY])
        echo ""
        
        wallet_name="default"
        hotkey_name="default"

        ohai "Creating new wallet. "
        echo ""
        echo ""
        ohai ">>> REMEMBER TO SAVE THE MNEMONICS <<<"
        wait_for_user 

        echo ""
        echo ""
        ohai "Creating wallet coldkey..."
        bittensor-cli new_coldkey --wallet.name $wallet_name
        RESULT=$?

        if [ $RESULT -eq 0 ]; then
          echo ""
          ohai "Wallet coldkey created successfully"
        fi

        echo ""
        echo ""
        ohai "Creating wallet hotkey..."
        bittensor-cli new_hotkey --wallet.name $wallet_name --wallet.hotkey $hotkey_name
        RESULT=$?

        if [ $RESULT -eq 0 ]; then
          echo ""
          ohai "Wallet hotkey created successfully"
          echo "python3  ~/.bittensor/bittensor/miners/text/template_miner.py" >> ~/.bittensor/bittensor/scripts/run.sh
          
          # Make run.sh executable
          chmod +x ~/.bittensor/bittensor/scripts/run.sh

          # Create alias for quick run
          OS="$(uname)"
          if [[ "$OS" == "Linux" ]]; then
            echo "alias run_bittensor=\"/bin/bash -c ~/.bittensor/bittensor/scripts/run.sh\"" >> ~/.bashrc
          elif [[ "$OS" == "Darwin" ]]; then
            echo "alias run_bittensor=\"/bin/bash -c ~/.bittensor/bittensor/scripts/run.sh\"" >> ~/.bash_profile
          else
            abort "Bittensor is only supported on macOS and Linux"
          fi
        fi
      break
      ;;
          [nN][oO]|[nN])
      echo "No"
      break
              ;;
          *)
      echo "Invalid input..."
      ;;
      esac
    done
}


# Do install.
OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then

    which -s apt
    if [[ $? == 0 ]] ; then
        abort "This linux based install requires apt. To run with other distros (centos, arch, etc), you will need to manually install the requirements"
    fi
    echo """
    
██████╗░██╗████████╗████████╗███████╗███╗░░██╗░██████╗░█████╗░██████╗░
██╔══██╗██║╚══██╔══╝╚══██╔══╝██╔════╝████╗░██║██╔════╝██╔══██╗██╔══██╗
██████╦╝██║░░░██║░░░░░░██║░░░█████╗░░██╔██╗██║╚█████╗░██║░░██║██████╔╝
██╔══██╗██║░░░██║░░░░░░██║░░░██╔══╝░░██║╚████║░╚═══██╗██║░░██║██╔══██╗
██████╦╝██║░░░██║░░░░░░██║░░░███████╗██║░╚███║██████╔╝╚█████╔╝██║░░██║
╚═════╝░╚═╝░░░╚═╝░░░░░░╚═╝░░░╚══════╝╚═╝░░╚══╝╚═════╝░░╚════╝░╚═╝░░╚═╝
                                                    
                                                    - Mining a new element.
    """
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
    linux_update_pip
    linux_install_bittensor
    echo ""
    echo ""
    echo "######################################################################"
    echo "##                                                                  ##"
    echo "##                      BITTENSOR SETUP                             ##"
    echo "##                                                                  ##"
    echo "######################################################################"
    echo ""
    echo ""
    setup_wallet_and_miner

elif [[ "$OS" == "Darwin" ]]; then
    echo """
    
██████╗░██╗████████╗████████╗███████╗███╗░░██╗░██████╗░█████╗░██████╗░
██╔══██╗██║╚══██╔══╝╚══██╔══╝██╔════╝████╗░██║██╔════╝██╔══██╗██╔══██╗
██████╦╝██║░░░██║░░░░░░██║░░░█████╗░░██╔██╗██║╚█████╗░██║░░██║██████╔╝
██╔══██╗██║░░░██║░░░░░░██║░░░██╔══╝░░██║╚████║░╚═══██╗██║░░██║██╔══██╗
██████╦╝██║░░░██║░░░░░░██║░░░███████╗██║░╚███║██████╔╝╚█████╔╝██║░░██║
╚═════╝░╚═╝░░░╚═╝░░░░░░╚═╝░░░╚══════╝╚═╝░░╚══╝╚═════╝░░╚════╝░╚═╝░░╚═╝
                                                    
                                                    - Mining a new element.
    """
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
    mac_update_pip
    mac_install_bittensor
    echo ""
    echo ""
    echo "######################################################################"
    echo "##                                                                  ##"
    echo "##                      BITTENSOR SETUP                             ##"
    echo "##                                                                  ##"
    echo "######################################################################"
    setup_wallet_and_miner

else
  abort "Bittensor is only supported on macOS and Linux"
fi

# Use the shell's audible bell.
if [[ -t 1 ]]; then
printf "\a"
fi

echo ""
echo ""
ohai "Installation successful!"
echo ""
ohai "If you simply intend to mine, then restart your machine and call \"run_bittensor\"."
ohai "If you are a power user, you can just run the Python mining script directly: python3  ~/.bittensor/bittensor/miners/text/template_miner.py."
ohai "If you wish to build your own Bittensor model, refer to documentation. The template_miner serves as a launching point to do this."
echo ""
ohai "Follow-up:"
echo ""
echo "- Check your tao balance: "
echo "    $ bittensor-cli overview --wallet.name <your wallet name> --wallet.hotkey <your hotkey name> --subtensor.network akatsuki"
echo ""
ohai "Extras:"
echo ""
echo "- Read the docs: "
echo "    ${tty_underline}https://app.gitbook.com/@opentensor/s/bittensor/${tty_reset}"
echo ""
echo "- Visit our website: "
echo "    ${tty_underline}https://www.bittensor.com${tty_reset}"
echo ""
echo "- Join the discussion: "
echo "    ${tty_underline}https://discord.gg/3rUr6EcvbB${tty_reset}"
echo ""

    
