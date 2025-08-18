#!/bin/bash
# install_libs.sh - helper for Makefile to install required libraries

set -e

install_ubuntu_debian() {
    echo "Installing libraries on Debian/Ubuntu..."
    sudo apt update
    sudo apt install -y build-essential cmake pkg-config fftw3 libfftw3-dev libsndfile1-dev libpng-dev libopenblas-dev
}

install_fedora_redhat() {
    echo "Installing libraries on Fedora/RedHat..."
    sudo dnf install -y gcc gcc-c++ make cmake pkgconfig fftw-devel libsndfile-devel libpng-devel openblas-devel
}

install_arch() {
    echo "Installing libraries on Arch Linux..."
    sudo pacman -Syu --needed base-devel fftw libsndfile libpng openblas
}

install_macos() {
    echo "Installing libraries on macOS..."
    if ! command -v brew &>/dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
    brew install fftw libsndfile libpng openblas
}

detect_platform_and_install() {
    OS_TYPE="$(uname -s)"
    case "$OS_TYPE" in
        Linux)
            if [ -f /etc/debian_version ]; then
                install_ubuntu_debian
            elif [ -f /etc/fedora-release ] || [ -f /etc/redhat-release ]; then
                install_fedora_redhat
            elif [ -f /etc/arch-release ]; then
                install_arch
            else
                echo "Unsupported Linux distribution. Please install required libraries manually."
                exit 1
            fi
            ;;
        Darwin)
            install_macos
            ;;
        *)
            echo "Unsupported OS: $OS_TYPE"
            exit 1
            ;;
    esac
}

# Export function for Makefile usage
detect_platform_and_install
