pycforest
=====================================

This is a wrapper for cforests within the R party package, including variable importance methods from
the R package edarf.

See: 
https://cran.r-project.org/web/packages/party/party.pdf
https://cran.r-project.org/web/packages/edarf/edarf.pdf

Requirements 
(1) R
(2) R package "party"
(3) R package "edarf"
(4) Python rpy2 library

How to install requirements on Ubuntu 18.04
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9
sudo add-apt-repository 'deb https://cloud.r-project.org/bin/linux/ubuntu bionic-cran35/'
sudo apt update
sudo apt install r-base
sudo R -e 'install.packages(c("party", "edarf"))'
sudo pip3 install rpy2


Install via:
sudo pip3 install git+https://github.com/gavin-s-smith/pycforest.git


TODO: Make the wrapper sklearn compliant so it can be used with pdpbox