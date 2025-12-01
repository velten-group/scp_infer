#!/bin/bash
mkdir -p test
cd test
git clone git@github.com:jan-spr/inference-pkg.git
sudo docker-compose run -p 7777:8888 scp_infer_test_2 /bin/bash

