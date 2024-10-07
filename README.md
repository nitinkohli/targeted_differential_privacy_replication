# Tageted Differential Privacy Replication Code

## Introduction

This repository contains replication code for the paper "Enabling Humantiarian Applications with Targeted Differential Privacy" by Nitin Kohli and Joshua Blumenstock. This README provides information about the code structure and the analyss performed in the manuscript.

## System Requirements

## Installation Guide

## Demo and Instructions for Use

The mobile phone datasets from Togo and Nigeria contain detailed information on billions of mobile phone transactions. These data contain proprietary and confidential information belonging to a private telecommunications operator and cannot be publicly released. 

As such, for this demo we provide a sample input dataset in the `Data` folder called `replication_input.csv`. 

To get the accuracy metrics for our private projection approach on the Demo data, naviate to `Scripts/prediction_measurements` and run:

`python prediction_metrics.py`

This should run in less than 1 minute. The expected outputs can be found in `Outputs/prediction_metrics`.

To get the privacy metrics for our private projection approach on the Demo data, naviate to `Scripts/privacy_measurements` and run the following two commands:

`python measure_attribute_inference`

`python measure_pso.py`

The first scripts runs in ~1 minute, whereas the second is more computationally intenstive and runs in 10 mins. The expected outputs can be found in `Outputs/privacy_metrics`.


