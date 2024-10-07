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

[comment]: <> Next, install the following from github. We use this as a point-of-comparison for the accuracy that existing privacy-enhancing technologies can offer.
[comment]: <> - `mondrian` via https://github.com/Andrew0133/Mondrian-k-anonimity

## Installation Guide

After installing the necessary packages, the code will run out of the box (using the data files located in the data subfolder). The install-time should be near instantenous. Outputs will be written to the outputs subfolder, divided into output folders for `prediction_metrics` and `privacy_metrics`. 

## Demo and Instructions for Use

### Accuracy Metrics

To get the accuracy metrics for our private projection approach on the Demo data (as well as the baseline level of accuracy), naviate to `Scripts/prediction_measurements` and run:

`python prediction_metrics.py`

This should run in less than 1 minute. The expected outputs can be found in `Outputs/prediction_metrics`.

### Privacy Metrics

To get the privacy metrics for our private projection approach on the Demo data (as well as the baseline level of accuracy), naviate to `Scripts/privacy_measurements` and run the following two commands:

`python measure_attribute_inference`

`python measure_pso.py`

The first scripts runs in ~1 minute, whereas the second is more computationally intenstive and runs in 10 mins. The expected outputs can be found in `Outputs/privacy_metrics`.


