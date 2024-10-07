# Tageted Differential Privacy Replication Code

## Introduction

This repository contains replication code for the paper "Enabling Humantiarian Applications with Targeted Differential Privacy" by Nitin Kohli and Joshua Blumenstock (https://arxiv.org/pdf/2408.13424). This README provides information about the code structure and the analyss performed in the manuscript.

As noted in our manuscript, the mobile phone datasets from Togo and Nigeria contain detailed information on billions of mobile phone transactions. These data contain proprietary and confidential information belonging to a private telecommunications operator and cannot be publicly released. Consequently, for this demo we provide a sample input dataset in the `Data` folder called `replication_input.csv`.

## System Requirements

Thie demo was developed and tested using Python 3.11.5 on macOS 14.6.1. 

To use this demo, first `pip install` the following packages.

- `numpy` (1.24.3)
- `pandas` (2.0.3)
- `anonymeter`

<!-- Next, install the following from github. We use this as a point-of-comparison for the accuracy that existing privacy-enhancing technologies can offer. -->
<!-- - `mondrian` via https://github.com/Andrew0133/Mondrian-k-anonimity -->

## Installation Guide

This repo can be installed via 

`git clone https://github.com/nitinkohli/targeted_differential_privacy_replication/` 

The install-time should be near instantenous.

## Demo and Instructions for Use

After installing the necessary packages, the code will run out of the box (using the data files located in the data subfolder). Outputs will be written to the outputs subfolder, divided into output folders for `prediction_metrics` and `privacy_metrics`. 

### Accuracy Metrics

To get the accuracy metrics for our private projection approach on the Demo data (as well as the baseline level of accuracy), naviate to `scripts/prediction_measurements` and run:

`python prediction_metrics.py`

This should run in less than 1 minute. The expected outputs can be found in `outputs/prediction_metrics`. At the top of this script, example parameters are set. For this demo, we only use a subset of the parameters considered in our paper to provide examples of how to code runs, and it's expected output.

### Privacy Metrics

To get the privacy metrics for our private projection approach on the Demo data (as well as the baseline level of accuracy), naviate to `scripts/privacy_measurements` and run the following two commands:

`python measure_attribute_inference`

`python measure_pso.py`

The first scripts runs in ~1 minute, whereas the second is more computationally intenstive and runs in ~10 mins. The expected outputs can be found in `outputs/privacy_metrics`. Similar to accuracy metrics script, example parameters are set at the top of each of these scripts to provide examples of the codes behavior and expected outputs.



